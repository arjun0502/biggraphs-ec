#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    // execute code below in parallel, with multiple threads
    #pragma omp parallel 
    {
        // initialize new frontier for each thread

        vertex_set thread_new_frontier;
        vertex_set_init(&thread_new_frontier, g->num_nodes);

        // process all vertices of current frontier in parallel
        // 
        #pragma omp for schedule(static) nowait  // the "nowait" clause allows threads to continue past the for loop and not wait for other threads to complete
        for (int i = 0; i < frontier->count; i++) {
            int vertex = frontier->vertices[i];

            int num_edges = outgoing_size(g, vertex);
            const int* outgoing = g->outgoing_edges + g->outgoing_starts[vertex];
            // process all outgoing vertices for current vertex 
            for (int j = 0; j < num_edges; j++) {
                int out_vertex = outgoing[j];                
                // use compare-and-swap to atomically check if outoging vertex is not visited and then update distance to outgoing vertex
                // do regular read first to avoid compare and swap
                if (distances[out_vertex] == NOT_VISITED_MARKER && __sync_bool_compare_and_swap(&distances[out_vertex], NOT_VISITED_MARKER, distances[vertex] + 1)) {
                    // add outgoing vertex to new frontier of specific thread
                    thread_new_frontier.vertices[thread_new_frontier.count++] = out_vertex; 
                }
            }
        }

        // merge new thread frontier into the global new frontier

        // use fetch_and_add() to atomically increment # of vertices in global new frontier by # of vertices in thread's new frontier and fetch the old # of vertices in the global new frontier
        int old_count = __sync_fetch_and_add(&new_frontier->count, thread_new_frontier.count);
        // Use the old value as the starting position for copying in thread's new frontier into global new frontier
        memcpy(new_frontier->vertices + old_count,
            thread_new_frontier.vertices, 
            thread_new_frontier.count * sizeof(int));
    }
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step_sequential(
    Graph g, 
    vertex_set* frontier, 
    vertex_set* new_frontier, 
    int* distances
) {
    // create bitmap that indicates whether each vertex in graph is in current frontier
    // index is id of vertex and value is 0 or 1 depending on if in frontier
    int* frontier_bitmap = (int*)calloc(g->num_nodes, sizeof(int));
    for (int i = 0; i < frontier->count; i++) {
        frontier_bitmap[frontier->vertices[i]] = 1;
    }

    // process all vertices of graph
    for (int v = 0; v < g->num_nodes; v++) {
        // skip vertex if already visited
        if (distances[v] != NOT_VISITED_MARKER) 
            continue;

        // process all incoming vertices for current vertex

        int num_edges = incoming_size(g, v);
        const int* incoming_edges = g->incoming_edges + g->incoming_starts[v];
        for (int j = 0; j < num_edges; j++) {
            int incoming = incoming_edges[j];

            // check if incoming vertex is in frontier
            if (frontier_bitmap[incoming]) {
                // mark current vertex as discovered and add to new frontier
                distances[v] = distances[incoming] + 1;
                new_frontier->vertices[new_frontier->count++] = v;
                break;  // each vertex is added only once
            }
        }        
    }

    free(frontier_bitmap);
}

void bottom_up_step_parallel(
    Graph g, 
    vertex_set* frontier, 
    vertex_set* new_frontier, 
    int* distances
) {
    // create bitmap that indicates whether each vertex in graph is in current frontier
    // index is id of vertex and value is 0 or 1 depending on if in frontier
    // do this in parallel using static scheduling since 
    int* frontier_bitmap = (int*)calloc(g->num_nodes, sizeof(int));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < frontier->count; i++) {
        frontier_bitmap[frontier->vertices[i]] = 1;
    }

    // execute code below in parallel with multiple threads
    #pragma omp parallel 
    {
        // initialize new frontier for each thread
        vertex_set thread_new_frontier;
        vertex_set_init(&thread_new_frontier, g->num_nodes);

        // process all vertices of graph in parallel
        #pragma omp for schedule(static) nowait
        for (int v = 0; v < g->num_nodes; v++) {
            // skip vertex if already visited
            if (distances[v] != NOT_VISITED_MARKER) 
                continue;

            // process all incoming vertices for current vertex
            int num_edges = incoming_size(g, v);
            const int* incoming_edges = g->incoming_edges + g->incoming_starts[v];
            for (int j = 0; j < num_edges; j++) {
                int in_vertex = incoming_edges[j];

                // skip incoming vertex if not in frontier
                if (!frontier_bitmap[in_vertex]) {
                    continue;
                }
                // use compare-and-swap to atomically mark current vertex as visited 
                if (__sync_bool_compare_and_swap(&distances[v], NOT_VISITED_MARKER, distances[in_vertex] + 1)) {
                    // add current vertex to thread's new frontier
                    thread_new_frontier.vertices[thread_new_frontier.count++] = v;
                    break; // each vertex only added once
                }
            }
        }
        
        // merge new thread frontier into the global new frontier
        // only do it if thread actually found new vertices
        if (thread_new_frontier.count > 0) {
            // use fetch_and_add() to atomically increment # of vertices in global new frontier by # of vertices in thread's new frontier and fetch the old # of vertices in the global new frontier
            int old_count = __sync_fetch_and_add(&new_frontier->count, thread_new_frontier.count);
            // Use the old value as the starting position for copying in thread's new frontier into global new frontier
            memcpy(new_frontier->vertices + old_count,
                thread_new_frontier.vertices, 
                thread_new_frontier.count * sizeof(int));
        }
    }

    free(frontier_bitmap);
}


void bfs_bottom_up(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[0] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    frontier->count = 1;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);

        bottom_up_step_parallel(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // Initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // Thresholds for switching between top-down and bottom-up
    // These values can be tuned based on specific graph characteristics
    const float alpha = 0.1;  // Switch to bottom-up when frontier size > alpha * |V|
    const float beta = 0.1;   // Switch back to top-down when frontier size < beta * |V|

    bool using_top_down = true;  // Start with top-down approach

    while (frontier->count != 0) {
        vertex_set_clear(new_frontier);

        // Calculate frontier size as fraction of total vertices
        float frontier_fraction = (float)frontier->count / graph->num_nodes;

        // Decide which approach to use
        if (using_top_down && frontier_fraction > alpha) {
            // Switch to bottom-up when frontier becomes too large
            using_top_down = false;
        } else if (!using_top_down && frontier_fraction < beta) {
            // Switch back to top-down when frontier becomes small
            using_top_down = true;
        }

        // Use chosen approach
        if (using_top_down) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            bottom_up_step_parallel(graph, frontier, new_frontier, sol->distances);
        }

        // Swap frontier pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

