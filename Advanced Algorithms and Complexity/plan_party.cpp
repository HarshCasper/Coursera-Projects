#include <iostream>
#include <vector>
#include <algorithm>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace std;

struct Vertex {
	int weight;
	vector <int> children;
};
typedef vector<Vertex> Graph;
typedef vector<int> Sum;

Graph ReadTree() {
	int vertices_count;
	cin >> vertices_count;

	Graph tree(vertices_count);

	for (int i = 0; i < vertices_count; ++i)
		cin >> tree[i].weight;

	for (int i = 1; i < vertices_count; ++i) {
		int from, to, weight;
		cin >> from >> to;
		tree[from - 1].children.push_back(to - 1);
		tree[to - 1].children.push_back(from - 1);
	}

	return tree;
}

int dfs(const Graph &tree, int v, int parent, vector<int> &d)
{
	if (d[v] != numeric_limits<int>::max()) return d[v];

	int m1 = tree[v].weight;
	for (int u: tree[v].children) {
		if (u == parent) continue;
		for (int w: tree[u].children) {
			if (w == v) continue;
			m1 += dfs(tree, w, u, d);
		}
	}

	int m0 = 0;
	for (int u: tree[v].children) {
		if (u == parent) continue;
		m0 += dfs(tree, u, v, d);
	}

	d[v] = max(m0, m1);
	return d[v];
}

int MaxWeightIndependentTreeSubset(const Graph &tree)
{
	vector<int> d(tree.size(), numeric_limits<int>::max());

	if (tree.empty()) return 0;

	return dfs(tree, 0, -1, d);
}

int main()
{
	// This code is here to increase the stack size to avoid stack overflow
	// in depth-first search.
	const rlim_t kStackSize = 64L * 1024L * 1024L;  // min stack size = 64 Mb
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur = kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}

	// Here begins the solution
	Graph tree = ReadTree();
	int weight = MaxWeightIndependentTreeSubset(tree);
	cout << weight << endl;
	return 0;
}
