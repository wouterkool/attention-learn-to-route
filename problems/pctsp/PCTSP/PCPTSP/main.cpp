#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#include <iterator>
#include <algorithm>

using namespace std;

struct Params {
	vector<vector<unsigned int>> vertices;
	vector<vector<unsigned int>> distanceMatrix;
	vector<bool> visited;
	unsigned int minTotalPrize;
	vector<vector<unsigned int>> orderSwapTwoOpt;

	Params() {
		minTotalPrize = 0;
	}
};

struct Solution {
	vector<unsigned int> route;
	double cost;
	double penalty;
	unsigned int prize;

	Solution() {
		cost = numeric_limits<double>::infinity();
		penalty = numeric_limits<double>::infinity();
		prize = 0;
	}
};

vector<string> openFile(unsigned int size, unsigned int maxPrize, unsigned int maxPenalty, unsigned int maxCost) {

	vector<string> lines;
	string file = "../Instances/problem_" + to_string(size) + "_" + to_string(maxPrize) + "_" + to_string(maxPenalty) + "_" + to_string(maxCost) + ".pctsp";
	ifstream inFile(file);

	if (inFile.good()) {
		string sLine;

		unsigned int j = 0;
		while (getline(inFile, sLine)) {
			if (sLine.length() != 0 && sLine != "\r") {
				lines.push_back(sLine);
				j++;
			}
		}
	}
	inFile.close();
	return lines;
}

vector<string> openFileName(string file) {

	vector<string> lines;
	ifstream inFile(file);

	if (inFile.good()) {
		string sLine;

		unsigned int j = 0;
		while (getline(inFile, sLine)) {
			if (sLine.length() != 0 && sLine != "\r") {
				lines.push_back(sLine);
				j++;
			}
		}
	}
	inFile.close();
	return lines;
}

void createGraph(const vector<string> &lines, Params *params) {

	istringstream iss(lines[0]);
	vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
	for (unsigned int i = 0; i < tokens.size(); i++) {
		params->vertices.push_back({ (unsigned int)stoi(tokens[i]) });
	}

	istringstream iss2(lines[1]);
	vector<string> tokens2{ istream_iterator<string>{iss2}, istream_iterator<string>{} };
	for (unsigned int i = 0; i < tokens2.size(); i++) {
		params->vertices[i].push_back((unsigned int)stoi(tokens2[i]));
	}

	for (unsigned int i = 2; i < lines.size(); i++) {
		params->distanceMatrix.emplace_back();
		istringstream iss3(lines[i]);
		vector<string> tokens3{ istream_iterator<string>{iss3}, istream_iterator<string>{} };
		for (unsigned int j = 0; j < tokens.size(); j++) {
			params->distanceMatrix[i - 2].push_back((unsigned int)stoi(tokens3[j]));
		}
	}
}

int genRandom(int i) {
	return std::rand() % i;
}

void randomConst(Params *params, Solution *solution) {

	vector<unsigned int> selected;
	for (unsigned int i = 1; i < params->vertices.size(); i++) {
		params->visited[i] = false;
		selected.push_back(i);
	}
	random_shuffle(selected.begin(), selected.end(), genRandom);

	solution->cost = 0;
	solution->penalty = 0;
	solution->route.push_back(0);
	params->visited[0] = true;
	unsigned int i = 0;
	while (solution->prize < params->minTotalPrize) {
		solution->cost += params->distanceMatrix[solution->route.back()][selected[i]];
		solution->prize += params->vertices[selected[i]][0];
		params->visited[selected[i]] = true;
		solution->route.push_back(selected[i]);
		i++;
	}
	solution->cost += params->distanceMatrix[solution->route.back()][0];
	solution->route.push_back(0);
	for (unsigned int i = 0; i < params->visited.size(); i++) {
		if (!params->visited[i])
			solution->penalty += params->vertices[i][1];
	}
}

void greedyRandomizedConst(Params *params, Solution *solution) {

	double minCost, currentCost;
	unsigned int solutionSize, selectedPos;
	auto *solutionCandidate = new Solution();
	vector<unsigned int> rlc;

	for (unsigned int i = 0; i < 10; i++) {
		solutionCandidate->prize = 0;
		solutionCandidate->cost = 0;
		solutionCandidate->route = {};
		solutionCandidate->route.push_back(0);
		solutionSize = (unsigned int)((i + 1) * params->vertices.size() / 10);
		params->visited[0] = true;
		for (unsigned int j = 1; j < params->vertices.size(); j++)
			params->visited[j] = false;

		while (solutionCandidate->prize < params->minTotalPrize || solutionCandidate->route.size() < solutionSize) {
			minCost = numeric_limits<double>::infinity();
			for (unsigned int j = 0; j < params->vertices.size(); j++) {
				if (!params->visited[j]) {
					currentCost = params->distanceMatrix[solutionCandidate->route.back()][j];
					if (currentCost < minCost)
						minCost = currentCost;
				}
			}
			rlc = {};
			for (unsigned int j = 0; j < params->vertices.size(); j++) {
				if (!params->visited[j] && (params->distanceMatrix[solutionCandidate->route.back()][j] <= 1.2 * minCost))
					rlc.push_back(j);
			}

			selectedPos = (unsigned int)(rand() % rlc.size());
			selectedPos = rlc[selectedPos];
			params->visited[selectedPos] = true;
			solutionCandidate->prize += params->vertices[selectedPos][0];
			solutionCandidate->cost += params->distanceMatrix[solutionCandidate->route.back()][selectedPos];
			solutionCandidate->route.push_back(selectedPos);
		}

		solutionCandidate->penalty = 0;
		for (unsigned int j = 0; j < params->visited.size(); j++) {
			if (!params->visited[j])
				solutionCandidate->penalty += params->vertices[j][1];
		}

		solutionCandidate->cost += params->distanceMatrix[solutionCandidate->route.back()][0];
		solutionCandidate->route.push_back(0);
		if (solutionCandidate->cost + solutionCandidate->penalty < solution->cost + solution->penalty) {
			solution->cost = solutionCandidate->cost;
			solution->penalty = solutionCandidate->penalty;
			solution->prize = solutionCandidate->prize;
			solution->route = solutionCandidate->route;
		}
	}
	delete solutionCandidate;
}

bool addNode(Params *params, Solution *solution, Solution *bestSolution) {
	double modifiedCost, modifiedPenalty;
	unsigned int modifiedPrize;
	int iBest = -1, jBest = -1;

	for (unsigned int i = 1; i < params->vertices.size() - 1; i++) {
		if (!params->visited[i]) {
			for (unsigned int j = 1; j < solution->route.size(); j++) {
				modifiedCost = solution->cost - params->distanceMatrix[solution->route[j - 1]][solution->route[j]] +
					params->distanceMatrix[solution->route[j - 1]][i] + params->distanceMatrix[i][solution->route[j]];
				modifiedPenalty = solution->penalty - params->vertices[i][1];
				modifiedPrize = solution->prize + params->vertices[i][0];

				if ((modifiedCost + modifiedPenalty) < (bestSolution->cost + bestSolution->penalty)) {
					iBest = i;
					jBest = j;
					bestSolution->cost = modifiedCost;
					bestSolution->penalty = modifiedPenalty;
					bestSolution->prize = modifiedPrize;
				}
			}
		}
	}
	if (iBest != -1) {
		solution->cost = bestSolution->cost;
		solution->penalty = bestSolution->penalty;
		solution->prize = bestSolution->prize;
		solution->route.insert(solution->route.begin() + jBest, (unsigned int)iBest);
		params->visited[iBest] = true;
		return true;
	}
	return false;
}

bool removeNode(Params *params, Solution *solution, Solution *bestSolution) {

	double modifiedCost, modifiedPenalty;
	unsigned int modifiedPrize;
	int iBest = -1;

	for (unsigned int i = 1; i < solution->route.size() - 1; i++) {
		modifiedCost = solution->cost - params->distanceMatrix[solution->route[i - 1]][solution->route[i]] -
			params->distanceMatrix[solution->route[i]][solution->route[i + 1]] +
			params->distanceMatrix[solution->route[i - 1]][solution->route[i + 1]];
		modifiedPenalty = solution->penalty + params->vertices[solution->route[i]][1];
		modifiedPrize = solution->prize - params->vertices[solution->route[i]][0];

		if ((modifiedPrize >= params->minTotalPrize) && (modifiedCost + modifiedPenalty < bestSolution->cost + bestSolution->penalty)) {
			iBest = i;
			bestSolution->cost = modifiedCost;
			bestSolution->penalty = modifiedPenalty;
			bestSolution->prize = modifiedPrize;
		}
	}
	if (iBest != -1) {
		solution->cost = bestSolution->cost;
		solution->penalty = bestSolution->penalty;
		solution->prize = bestSolution->prize;
		params->visited[solution->route[iBest]] = false;
		solution->route.erase(solution->route.begin() + iBest);
		return true;
	}
	return false;
}

bool swapNodes(Params *params, Solution *solution, Solution *bestSolution) {

	double modifiedCost;
	int iBest = -1, jBest = -1;

	//    for (const vector<unsigned int> &pos: params->orderSwapTwoOpt) {
	for (unsigned int i = 1; i < solution->route.size() - 1; i++) {
		for (unsigned int j = i + 1; j < solution->route.size() - 1; j++) {
			modifiedCost = solution->cost - params->distanceMatrix[solution->route[i - 1]][solution->route[i]] -
				params->distanceMatrix[solution->route[j]][solution->route[j + 1]] +
				params->distanceMatrix[solution->route[i - 1]][solution->route[j]] +
				params->distanceMatrix[solution->route[j]][solution->route[i + 1]] +
				params->distanceMatrix[solution->route[j - 1]][solution->route[i]] +
				params->distanceMatrix[solution->route[i]][solution->route[j + 1]];

			if (j != i + 1)
				modifiedCost = modifiedCost - params->distanceMatrix[solution->route[i]][solution->route[i + 1]] -
				params->distanceMatrix[solution->route[j - 1]][solution->route[j]];

			if (modifiedCost < bestSolution->cost) {
				iBest = i;
				jBest = j;
				bestSolution->cost = modifiedCost;
			}
		}
	}

	if (iBest != -1) {
		solution->cost = bestSolution->cost;
		unsigned int temp;
		temp = solution->route[(unsigned int)iBest];
		solution->route[(unsigned int)iBest] = solution->route[(unsigned int)jBest];
		solution->route[(unsigned int)jBest] = temp;
		return true;
	}
	return false;
}

bool twoOpt(Params *params, Solution *solution, Solution *bestSolution) {

	double modifiedCost;
	int iBest = -1, jBest = -1;

	//    for (const vector<unsigned int> &pos: params->orderSwapTwoOpt) {
	for (unsigned int i = 1; i < solution->route.size() - 1; i++) {
		for (unsigned int j = i + 1; j < solution->route.size() - 1; j++) {
			modifiedCost = solution->cost - params->distanceMatrix[solution->route[i - 1]][solution->route[i]] -
				params->distanceMatrix[solution->route[j]][solution->route[j + 1]] +
				params->distanceMatrix[solution->route[i - 1]][solution->route[j]] +
				params->distanceMatrix[solution->route[i]][solution->route[j + 1]];

			if (modifiedCost < bestSolution->cost) {
				iBest = i;
				jBest = j;
				bestSolution->cost = modifiedCost;
			}
		}
	}

	if (iBest != -1) {
		solution->cost = bestSolution->cost;
		bestSolution->route = {};
		copy(solution->route.begin() + iBest, solution->route.begin() + jBest + 1,
			back_inserter(bestSolution->route));
		reverse(bestSolution->route.begin(), bestSolution->route.end());
		solution->route.erase(solution->route.begin() + iBest, solution->route.begin() + jBest + 1);
		solution->route.insert(solution->route.begin() + iBest, bestSolution->route.begin(), bestSolution->route.end());
		return true;
	}

	return false;
}

void orderLS(Params *params, Solution *solution) {
	params->orderSwapTwoOpt = {};
	for (unsigned int i = 1; i < solution->route.size() - 1; i++) {
		for (unsigned int j = i + 1; j < solution->route.size() - 1; j++) {
			params->orderSwapTwoOpt.push_back({ i, j });
		}
	}
}

void shuffleIndices(Params *params) {
	// Shuffling the jobs order vector
	random_shuffle(params->orderSwapTwoOpt.begin(), params->orderSwapTwoOpt.end(), genRandom);
}

void localSearch(Params *params, Solution *solution) {

	bool foundBetter1, foundBetter2, foundBetter3, foundBetter4;
	auto *bestSolution = new Solution();
	unsigned int improved = 0;

	for (unsigned int i = 0; i < params->visited.size(); i++)
		params->visited[i] = false;
	for (unsigned int i = 0; i < solution->route.size(); i++) {
		params->visited[solution->route[i]] = true;
	}
	bestSolution->cost = solution->cost;
	bestSolution->penalty = solution->penalty;
	bestSolution->prize = solution->prize;

	while (true) {
		foundBetter1 = addNode(params, solution, bestSolution);
		//        orderLS(params, solution);
		//        shuffleIndices(params);
		foundBetter2 = swapNodes(params, solution, bestSolution);
		foundBetter3 = removeNode(params, solution, bestSolution);
		//        orderLS(params, solution);
		//        shuffleIndices(params);
		foundBetter4 = twoOpt(params, solution, bestSolution);
		if (!foundBetter1 && !foundBetter2 && !foundBetter3 && !foundBetter4)
			break;
		improved++;
	}
	//    cout << improved << endl;
	delete bestSolution;
}

void doubleBridge(Params *params, Solution *solutionCandidate) {

	solutionCandidate->route.pop_back();

	unsigned int position1 = 1 + (unsigned int)(rand() % (int)(solutionCandidate->route.size() / 3));
	unsigned int position2 = position1 + 1 + (unsigned int)(rand() % (int)(solutionCandidate->route.size() / 3));
	unsigned int position3 = position2 + 1 + (unsigned int)(rand() % (int)(solutionCandidate->route.size() / 3));

	vector<unsigned int> temp = {};

	copy(solutionCandidate->route.begin(), solutionCandidate->route.begin() + position1, back_inserter(temp));
	temp.insert(temp.end(), solutionCandidate->route.begin() + position3, solutionCandidate->route.end());
	temp.insert(temp.end(), solutionCandidate->route.begin() + position2, solutionCandidate->route.begin() + position3);
	temp.insert(temp.end(), solutionCandidate->route.begin() + position1, solutionCandidate->route.begin() + position2);

	solutionCandidate->route = temp;
	solutionCandidate->route.push_back(0);
	solutionCandidate->cost = 0;
	for (unsigned int i = 0; i < solutionCandidate->route.size() - 1; i++)
		solutionCandidate->cost += params->distanceMatrix[solutionCandidate->route[i]][solutionCandidate->route[i + 1]];
}

void perturbation(Params *params, Solution *solutionCandidate, unsigned int intensity) {
    // Need at least 4 nodes / two internal nodes (0 1 2 0) to perform double bridge
    if (solutionCandidate->route.size() < 4)
        return;
	for (unsigned int i = 0; i < intensity; i++) {
		doubleBridge(params, solutionCandidate);
	}
}

void ILS(Params *params, Solution *solution) {
	auto *bestSolution = new Solution();
	bestSolution->route = solution->route;
	bestSolution->cost = solution->cost;
	bestSolution->penalty = solution->penalty;
	bestSolution->prize = solution->prize;

	auto *modifiedSolution = new Solution();
	modifiedSolution->route = solution->route;
	modifiedSolution->cost = solution->cost;
	modifiedSolution->penalty = solution->penalty;
	modifiedSolution->prize = solution->prize;

	unsigned int maxIter = 40000;
	unsigned int maxNoImprov = 20000;
	unsigned int maxReboot = 4001;
	unsigned int maxNbReboots = 4;
	unsigned int iterations = 0;
	unsigned int noImprov = 0;
	unsigned int reboot = 0;
	unsigned int nbReboots = 0;

	while (noImprov < maxNoImprov && iterations < maxIter) {

		perturbation(params, modifiedSolution, 2);
		localSearch(params, modifiedSolution);

		iterations++;
		noImprov++;
		reboot++;

		if (modifiedSolution->cost + modifiedSolution->penalty < solution->cost + solution->penalty) {
			//            cout << iterations << endl;
			//            cout << modifiedSolution->cost + modifiedSolution->penalty << endl;
			solution->route = modifiedSolution->route;
			solution->cost = modifiedSolution->cost;
			solution->penalty = modifiedSolution->penalty;
			solution->prize = modifiedSolution->prize;
			if (modifiedSolution->cost + modifiedSolution->penalty < bestSolution->cost + bestSolution->penalty) {
				bestSolution->route = modifiedSolution->route;
				bestSolution->cost = modifiedSolution->cost;
				bestSolution->penalty = modifiedSolution->penalty;
				bestSolution->prize = modifiedSolution->prize;
				reboot = 0;
				noImprov = 0;
			}
		}
		else {
			modifiedSolution->route = solution->route;
			modifiedSolution->cost = solution->cost;
			modifiedSolution->penalty = solution->penalty;
			modifiedSolution->prize = solution->prize;
		}

		if (nbReboots < maxNbReboots && reboot >= maxReboot) {
			modifiedSolution->cost = numeric_limits<double>::infinity();
			greedyRandomizedConst(params, modifiedSolution);
			localSearch(params, modifiedSolution);
			solution->route = modifiedSolution->route;
			solution->cost = modifiedSolution->cost;
			solution->penalty = modifiedSolution->penalty;
			solution->prize = modifiedSolution->prize;
			if (modifiedSolution->cost + modifiedSolution->penalty < bestSolution->cost + bestSolution->penalty) {
				bestSolution->route = modifiedSolution->route;
				bestSolution->cost = modifiedSolution->cost;
				bestSolution->penalty = modifiedSolution->penalty;
				bestSolution->prize = modifiedSolution->prize;
				noImprov = 0;
			}
			reboot = 0;
			nbReboots++;
		}
	}
	solution->route = bestSolution->route;
	solution->cost = bestSolution->cost;
	solution->penalty = bestSolution->penalty;
	solution->prize = bestSolution->prize;
	//    cout << iterations << endl;
	//    cout << solution->cost + solution->penalty << endl;
	//    cout << "--------" << endl;
	delete modifiedSolution;
	delete bestSolution;
}
//
//int main() {
//
//	unsigned int runs = 20;
//	vector<unsigned int> sizes = { 20, 40, 60, 80, 100, 200, 300, 400, 500 };
//	vector<unsigned int> penalties = { 100, 1000 };
//	vector<unsigned int> costs = { 1000, 10000 };
//	vector<string> lines;
//	double bestResult;
//	double averageResult, averageTime;
//
//	for (const unsigned int &size : sizes) {
//		for (const unsigned int &penalty : penalties) {
//			for (const unsigned int &cost : costs) {
//				if (!(penalty == 1000 && cost == 1000)) {
//					bestResult = numeric_limits<double>::infinity();
//					averageResult = 0;
//					averageTime = 0;
//					for (unsigned int k = 0; k < runs; k++) {
//						srand((unsigned int)time(nullptr));
//						const clock_t start = clock();
//						auto *params = new Params();
//
//						lines = openFile(size, 100, penalty, cost);
//						createGraph(lines, params);
//						for (unsigned int i = 0; i < params->vertices.size(); i++)
//							params->visited.push_back(false);
//
//						auto *solution = new Solution();
//						//                        randomConst(params, solution);
//						greedyRandomizedConst(params, solution);
//
//						//                        orderLS(params, solution);
//						localSearch(params, solution);
//						//                      for (unsigned int i = 0; i < solution->route.size(); i++)
//						//                          cout << solution->route[i] << ", ";
//						//                      cout << endl;
//						ILS(params, solution);
//
//						if (solution->cost + solution->penalty < bestResult)
//							bestResult = solution->cost + solution->penalty;
//						averageResult += (solution->cost + solution->penalty);
//						averageTime += (float(clock() - start) / CLOCKS_PER_SEC);
//
//						delete solution;
//						delete params;
//					}
//					cout << "Best Result: " << bestResult << endl;
//					cout << "Average Result: " << averageResult / runs << endl;
//					cout << "Average Time: " << averageTime / runs << endl;
//				}
//			}
//		}
//	}
//	return 0;
//}


int main(int argc, char *argv[]) {

    string filename = argv[1];
    unsigned int minTotalPrize = stoi(argv[2]);
	unsigned int runs = 20;
	unsigned int seed = 1234;
	if (argc >= 4)
    {
        runs = stoi(argv[3]);
    }
    if (argc >= 5)
    {
        seed = stoi(argv[4]);
    }

	vector<string> lines;
	double bestResult;
	double averageResult, averageTime;
	vector<unsigned int> bestRoute;

    bestResult = numeric_limits<double>::infinity();
    bestRoute = vector<unsigned int>();
    averageResult = 0;
    averageTime = 0;
    for (unsigned int k = 0; k < runs; k++) {
        cout << "Run: " << k << endl;
        srand((unsigned int)seed + k);
        const clock_t start = clock();
        auto *params = new Params();
        params->minTotalPrize = minTotalPrize;

        // lines = openFile(size, 100, penalty, cost);
        lines = openFileName(filename);
        createGraph(lines, params);
        for (unsigned int i = 0; i < params->vertices.size(); i++)
            params->visited.push_back(false);

        auto *solution = new Solution();
        //                        randomConst(params, solution);
        greedyRandomizedConst(params, solution);

        //                        orderLS(params, solution);
        localSearch(params, solution);
        //                      for (unsigned int i = 0; i < solution->route.size(); i++)
        //                          cout << solution->route[i] << ", ";
        //                      cout << endl;
        ILS(params, solution);

        if (solution->cost + solution->penalty < bestResult)
            bestResult = solution->cost + solution->penalty;
            bestRoute = solution->route;
        averageResult += (solution->cost + solution->penalty);
        averageTime += (float(clock() - start) / CLOCKS_PER_SEC);

        delete solution;
        delete params;
    }
    cout << "Best Result Cost: " << bestResult << endl;
    cout << "Best Result Route:";
    for (auto i: bestRoute)
        cout << ' ' << i;
    cout << endl;
    cout << "Average Result: " << averageResult / runs << endl;
    cout << "Average Time: " << averageTime / runs << endl;
	return 0;
}

