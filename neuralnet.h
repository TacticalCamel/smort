#pragma once

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <windows.h>

///general simulation settings
extern const int CPU_THREADS, DNA_SEGMENT_SIZE, BOARD_SIZE;
extern const int INPUT_NODES, OUTPUT_NODES, HIDDEN_NODES, NODE_COUNT;

extern const int GENERATIONS, POPULATION, SIMULATION_STEPS, DNA_SEGMENT_COUNT;
extern const double MUTATION_CHANCE;

///global variables
extern std::default_random_engine randomGen;
extern std::uniform_int_distribution<int> randInt;
extern int** occupiedPositions;

class Neuralnet{
private:
    ///private variables
    typedef enum { INPUT, HIDDEN, OUTPUT } NodeType;
    typedef struct { int x; int y; } Vector2;
    typedef struct { NodeType type; double value; } Node;
    typedef struct { double weight; Node* from; Node* to; } Link;

    ///private functions
    void changePosition(int x, int y);
    void setDna(std::string* inheritedDna);
    void setRandomPosition();
    void createNodes();
    void createLinks();

public:
    ///public variables
    Vector2 pos;
    int age;
    int linkCount;
    int forwardDir;
    Node* nodes;
    std::string* dna;
    Link* links;

    ///public functions
    Neuralnet();
    void reset(std::string* inheritedDna);
    void step();
};
