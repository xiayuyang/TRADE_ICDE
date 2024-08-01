#include "Matching.h"
#include <fstream>
#include "Graph.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
// 编译命令 g++ -o my_example my_Example.cpp Matching.cpp Graph.cpp BinaryHeap.cpp
Graph ReadGraph()
{
	string line;
	
	getline(cin, line);
	int n = stoi(line);
	getline(cin, line);
	int m = stoi(line);
	Graph G(n);
	while (getline(cin, line))
	{
		char discard;
		double dis_num;
		int u, v;
		istringstream ss(line);
		ss >> discard >> u >> discard >> v >> discard >> dis_num >> discard;
		G.AddEdge(u, v);
	}

	return G;
}

void MaximumMatchingExample()
{
	Graph G = ReadGraph();
	Matching M(G);

	list<int> matching;
	matching = M.SolveMaximumMatching();

	for(list<int>::iterator it = matching.begin(); it != matching.end(); it++)
	{
		pair<int, int> e = G.GetEdge( *it );

		cout << e.first << " " << e.second << endl;
	}
}

int main()
{

	try
	{
		MaximumMatchingExample();
	}
	catch(const char * msg)
	{
		cout << msg << endl;
		return 1;
	}

	return 0;
}



