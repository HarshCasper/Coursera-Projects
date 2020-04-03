#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <queue>

using namespace std;

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

class MaxMatching {
	public:
		void Solve()
		{
			vector<vector<bool>> adj_matrix = ReadData();
			vector<int> matching = FindMatching(adj_matrix);

			WriteResponse(matching);
		}

	private:
		vector<vector<bool>> ReadData()
		{
			int num_left, num_right;
			cin >> num_left >> num_right;
			vector<vector<bool>> adj_matrix(num_left, vector<bool>(num_right));

			for (int i = 0; i < num_left; ++i) {
				for (int j = 0; j < num_right; ++j) {
					int bit;
					cin >> bit;
					adj_matrix[i][j] = (bit == 1);
				}
			}

			return adj_matrix;
		}

		void WriteResponse(const vector<int>& matching)
		{
			for (int i = 0; i < matching.size(); ++i) {
				if (i > 0) {
					cout << " ";
				}
				if (matching[i] == -1) {
					cout << "-1";
				} else {
					cout << (matching[i] + 1);
				}
			}
			cout << "\n";
		}

		vector<int> FindMatching(const vector<vector<bool>>& adj_matrix)
		{
			vector<int> matching;
			int num_left = adj_matrix.size();
			int num_right = adj_matrix[0].size();
			int n = num_left + num_right + 2;
			FlowGraph graph(n);
			int source = 0;
			int sink = n - 1;

			for (int i = 0; i < num_left; i++) {
				graph.add_edge(source, i+1, 1);
			}

			for (int i = 0; i < num_left; i++) {
				for (int j = 0; j < num_right; j++) {
					if (adj_matrix[i][j]) {
						graph.add_edge(i+1, j+num_left+1, 1);
					}
				}
			}

			for (int j = 0; j < num_right; j++) {
				graph.add_edge(j+num_left+1, sink, 1);
			}

			max_flow(graph, source, sink);

			for (int i = 0; i < num_left; i++) {
				const vector<size_t> &edge_ids = graph.get_ids(i+1);
				int match_id = -1;

				for (size_t j = 0; j < edge_ids.size(); j++) {
					const FlowGraph::Edge &edge = graph.get_edge(edge_ids[j]);

					if (edge.to == source) continue;
					if (edge.flow <= 0) continue;

					match_id = edge.to-num_left-1;
				}

				matching.push_back(match_id);
			}

			return matching;
		}
};

int main()
{
	ios_base::sync_with_stdio(false);
	MaxMatching max_matching;

	max_matching.Solve();

	return 0;
}
