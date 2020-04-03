#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

/* This class implements a bit unusual scheme for storing edges of the graph,
 * in order to retrieve the backward edge for a given edge quickly. */
class FlowGraph {
	public:
		struct Edge {
			int from, to, capacity, flow;
		};

	private:
		/* List of all - forward and backward - edges */
		vector<Edge> edges;

		/* These adjacency lists store only indices of edges in the edges list */
		vector<vector<size_t> > graph;

	public:
		explicit FlowGraph(size_t n): graph(n) {}

		void add_edge(int from, int to, int capacity) {
			/* Note that we first append a forward edge and then a backward edge,
			 * so all forward edges are stored at even indices (starting from 0),
			 * whereas backward edges are stored at odd indices in the list edges */
			Edge forward_edge = {from, to, capacity, 0};
			Edge backward_edge = {to, from, 0, 0};
			graph[from].push_back(edges.size());
			edges.push_back(forward_edge);
			graph[to].push_back(edges.size());
			edges.push_back(backward_edge);
		}

		size_t size() const {
			return graph.size();
		}

		const vector<size_t>& get_ids(int from) const {
			return graph[from];
		}

		const Edge& get_edge(size_t id) const {
			return edges[id];
		}

		void add_flow(size_t id, int flow) {
			/* To get a backward edge for a true forward edge (i.e id is even), we should get id + 1
			 * due to the described above scheme. On the other hand, when we have to get a "backward"
			 * edge for a backward edge (i.e. get a forward edge for backward - id is odd), id - 1
			 * should be taken.
			 *
			 * It turns out that id ^ 1 works for both cases. Think this through! */
			edges[id].flow += flow;
			edges[id ^ 1].flow -= flow;
		}
};

static FlowGraph read_data()
{
	int vertex_count, edge_count;
	cin >> vertex_count >> edge_count;
	FlowGraph graph(vertex_count);
	for (int i = 0; i < edge_count; ++i) {
		int u, v, capacity;
		cin >> u >> v >> capacity;
		graph.add_edge(u - 1, v - 1, capacity);
	}

	/*
	FlowGraph graph(2);

	graph.add_edge(1 - 1, 1 - 1, 10000);
	graph.add_edge(1 - 1, 2 - 1, 1);
	graph.add_edge(1 - 1, 2 - 1, 4);
	graph.add_edge(1 - 1, 2 - 1, 100);
	graph.add_edge(2 - 1, 1 - 1, 900);
	*/

	return graph;
}

int max_flow(FlowGraph& graph, size_t from, size_t to)
{
	int ret = 0;
	bool found_path = true;

	while (found_path) {
		queue<size_t> q;
		vector<bool> visited(graph.size(), false);
		vector<size_t> previous(graph.size(), -1);
		int min_flow = numeric_limits<int>::max();

		found_path = false;

		q.push(from);
		visited[from] = true;

		while (!q.empty() && !found_path) {
			size_t x = q.front();
			q.pop();

			const vector<size_t> &edge_ids = graph.get_ids(x);

			for (size_t i = 0; i < edge_ids.size(); i++) {
				const FlowGraph::Edge &edge = graph.get_edge(edge_ids[i]);
				int f = edge.capacity - edge.flow;

				if (f <= 0) continue;

				size_t y = (x == edge.from) ? edge.to : edge.from;

				if (visited[y]) continue;
				visited[y] = true;
				q.push(y);
				previous[y] = x;

				if (y == to) {
					found_path = true;
					break;
				}
			}
		}

		if (found_path) {
			vector<size_t> path;
			size_t v = to;

			while (v != from) {
				size_t u = previous[v];
				const vector<size_t> &edge_ids = graph.get_ids(u);

				for (size_t i = 0; i < edge_ids.size(); i++) {
					const FlowGraph::Edge &edge = graph.get_edge(edge_ids[i]);
					int f = edge.capacity - edge.flow;

					if (f <= 0) continue;

					if (u == edge.from && v == edge.to) {
						min_flow = min(min_flow, f);
						path.push_back(edge_ids[i]);
						break;
					}
				}

				v = u;
			}

			for (size_t i = 0; i < path.size(); i++) {
				graph.add_flow(path[i], min_flow);
			}
			ret += min_flow;
		}
	}

	return ret;
}

int main()
{
	ios_base::sync_with_stdio(false);
	FlowGraph graph = read_data();

	cout << max_flow(graph, 0, graph.size() - 1) << "\n";
	return 0;
}
