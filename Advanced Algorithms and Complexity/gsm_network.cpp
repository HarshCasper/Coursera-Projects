#include <ios>
#include <iostream>
#include <vector>

using namespace std;

struct Edge {
	int from;
	int to;
};

struct ConvertGSMNetworkProblemToSat {
	int numVertices;
	vector<Edge> edges;

	ConvertGSMNetworkProblemToSat(int n, int m) :
		numVertices(n),
		edges(m)
	{}

	void printEquisatisfiableSatFormula()
	{
		int clauses_num = 4*numVertices + 3*edges.size();
		int vars_num = 3*numVertices;

		cout << clauses_num << " " << vars_num << endl;

		/* each node has only one color */
		for (int i = 0; i < 3*numVertices; i += 3) {
			cout << i+1 << " " << i+2 << " " << i+3 << " " << 0 << endl;
			cout << -(i+1) << " " << -(i+2) << " " << 0 << endl;
			cout << -(i+1) << " " << -(i+3) << " " << 0 << endl;
			cout << -(i+2) << " " << -(i+3) << " " << 0 << endl;
		}

		/* same edge nodes have different color */
		for (size_t i = 0; i < edges.size(); i++) {
			int from = 3*edges[i].from-2;
			int to = 3*edges[i].to-2;

			cout << -from << " " << -to << " " << 0 << endl;
			cout << -(from+1) << " " << -(to+1) << " " << 0 << endl;
			cout << -(from+2) << " " << -(to+2) << " " << 0 << endl;
		}
	}
};

int main()
{
	ios::sync_with_stdio(false);

	int n, m;
	cin >> n >> m;
	ConvertGSMNetworkProblemToSat converter(n, m);

	for (int i = 0; i < m; ++i) {
		cin >> converter.edges[i].from >> converter.edges[i].to;
	}

	converter.printEquisatisfiableSatFormula();

	return 0;
}
