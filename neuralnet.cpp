#include "neuralnet.h"

#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma ide diagnostic ignored "modernize-loop-convert"

///general simulation settings
const int CPU_THREADS = 12;
const int DNA_SEGMENT_SIZE = 6;
const int BOARD_SIZE = 100;
const int INPUT_NODES = 10;
const int OUTPUT_NODES = 10;

const int HIDDEN_NODES = 8;
const int NODE_COUNT = INPUT_NODES + OUTPUT_NODES + HIDDEN_NODES;

const int GENERATIONS = 2000;
const int POPULATION = 500;
const int SIMULATION_STEPS = 250;
const int DNA_SEGMENT_COUNT = 25;
const double MUTATION_CHANCE = 0.001;

///global variables
std::default_random_engine randomGen(time(nullptr));
std::uniform_int_distribution<int> randInt(0, INT_MAX);

///class functions
Neuralnet::Neuralnet(){
    nodes = new Node[NODE_COUNT];
    dna = new std::string[DNA_SEGMENT_COUNT];
    links = new Link[DNA_SEGMENT_COUNT];
}
void Neuralnet::changePosition(int x, int y){
    //move
    pos.x += x;
    pos.y += y;

    //outside board -> revert changes
    if(pos.x >= BOARD_SIZE || pos.x < 0){
        pos.x -= x;
        return;
    }
    if(pos.y >= BOARD_SIZE || pos.y < 0){
        pos.y -= y;
        return;
    }
    //already occupied -> revert changes
    if(occupiedPositions[pos.x][pos.y] == 1){
        pos.x -= x;
        pos.y -= y;
        return;
    }
    //valid move
    occupiedPositions[pos.x][pos.y] = 1;
    occupiedPositions[pos.x - x][pos.y - y] = 0;
}
void Neuralnet::setDna(std::string* inheritedDna){
    //random dna
    if(inheritedDna == nullptr){
        for(int i = 0; i < DNA_SEGMENT_COUNT; i++){
            dna[i].clear();
            for(int j = 0; j < DNA_SEGMENT_SIZE; j++){
                int x = (j == 0) ? randInt(randomGen) % (INPUT_NODES + HIDDEN_NODES) : (j == 1) ? randInt(randomGen) % (HIDDEN_NODES + OUTPUT_NODES) + INPUT_NODES : randInt(randomGen) % NODE_COUNT;
                dna[i].append(1, x < 10 ? (char) ('0' + x) : (char) ('A' + x - 10));
            }
        }
    }
    //inherited dna
    else{
        for(int i = 0; i < DNA_SEGMENT_COUNT; i++){
            dna[i] = inheritedDna[i];
            for(int j = 0; j < DNA_SEGMENT_SIZE; j++){
                if((double) randInt(randomGen) / INT_MAX < MUTATION_CHANCE){
                    int x = (j == 0) ? randInt(randomGen) % (INPUT_NODES + HIDDEN_NODES) : (j == 1) ? randInt(randomGen) % (HIDDEN_NODES + OUTPUT_NODES) + INPUT_NODES : randInt(randomGen) % NODE_COUNT;
                    dna[i][j] = x < 10 ? (char) ('0' + x) : (char) ('A' + x - 10);
                }
            }
        }
    }
}
void Neuralnet::setRandomPosition(){
    do{
        pos.x = randInt(randomGen) % BOARD_SIZE;
        pos.y = randInt(randomGen) % BOARD_SIZE;
    } while(occupiedPositions[pos.x][pos.y] == 1);
    occupiedPositions[pos.x][pos.y] = 1;
}
void Neuralnet::createNodes(){
    for(int i = 0; i < NODE_COUNT; i++){
        nodes[i].type = i < INPUT_NODES ? INPUT : (i < INPUT_NODES + HIDDEN_NODES ? HIDDEN : OUTPUT);
    }
}
void Neuralnet::createLinks(){
    for(int i = 0; i < DNA_SEGMENT_COUNT; i++){
        //get from and to ID
        int from = dna[i][0] <= '9' ? dna[i][0] - '0' : dna[i][0] - 'A' + 10;
        int to = dna[i][1] <= '9' ? dna[i][1] - '0' : dna[i][1] - 'A' + 10;

        //1. source cannot be OUTPUT type
        if(nodes[from].type == OUTPUT) continue;
        //2. destination cannot be INPUT type
        if(nodes[to].type == INPUT) continue;
        //3. no duplicate connections
        for(int j = 0; j < i; j++){
            if(dna[i][0] == dna[j][0] && dna[i][1] == dna[j][1]) continue;
        }

        //create link if fulfills requirements
        links[linkCount].from = &nodes[from];
        links[linkCount].to = &nodes[to];
        int maxVal = pow(NODE_COUNT, 4);
        links[linkCount].weight = 8.0 * strtol(dna[i].data() + 2, nullptr, NODE_COUNT) / maxVal - 4;
        linkCount++;
    }
}
void Neuralnet::reset(std::string* inheritedDna){
    age = 0;
    linkCount = 0;
    forwardDir = randInt(randomGen) % 4;

    setDna(inheritedDna);
    setRandomPosition();
    createNodes();
    createLinks();
}
void Neuralnet::step(){
    for(int i = 0; i < NODE_COUNT; i++){
        nodes[i].value = 0;
    }
    ///set input node values: [0; 1]
    //const 1
    nodes[0].value = 1;
    //x position
    nodes[1].value = (double) pos.x / BOARD_SIZE;
    //y position
    nodes[2].value = (double) pos.y / BOARD_SIZE;
    //age
    nodes[3].value = (double) (age++) / SIMULATION_STEPS;
    //random
    nodes[4].value = (double) randInt(randomGen) / INT_MAX;
    //forward direction
    nodes[5].value = forwardDir / 4.0;
    //nearest east/west border
    nodes[6].value = (BOARD_SIZE - abs(0.5 * BOARD_SIZE - pos.x)) / BOARD_SIZE;
    //nearest south/north border
    nodes[7].value = (BOARD_SIZE - abs(0.5 * BOARD_SIZE - pos.y)) / BOARD_SIZE;
    //population density nearby
    int pNear = 0;
    for(int i = -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            if(i == 0 && j == 0) continue;
            int x = pos.x + i;
            int y = pos.y + j;
            if(x < 0 || y < 0 || x >= BOARD_SIZE || y >= BOARD_SIZE) continue;
            if(occupiedPositions[x][y] == 1) pNear++;
        }
    }
    nodes[8].value = pNear / 8.0;
    //population density forward
    Vector2 d;
    d.x = forwardDir == 3 ? 1 : forwardDir == 1 ? -1 : 0;
    d.y = forwardDir == 0 ? 1 : forwardDir == 2 ? -1 : 0;
    int pForw = 0;
    for(int i = 1; i <= 4; i++){
        int x = pos.x + i * d.x;
        int y = pos.y + i * d.y;
        if(x < 0 || y < 0 || x >= BOARD_SIZE || y >= BOARD_SIZE) continue;
        if(occupiedPositions[x][y] == 1) pForw++;
    }
    nodes[9].value = pForw / 4.0;

    ///propagate through links
    //input -> any
    for(int i = 0; i < linkCount; i++){
        if(links[i].from->type == INPUT){
            links[i].to->value += links[i].from->value * links[i].weight;
        }
    }
    //hidden -> hidden
    for(int i = 0; i < linkCount; i++){
        if(links[i].from->type == HIDDEN && links[i].to->type == HIDDEN){
            links[i].to->value *= tanh(links[i].from->value) * links[i].weight;
        }
    }
    //hidden -> output
    for(int i = 0; i < linkCount; i++){
        if(links[i].from->type == HIDDEN){
            links[i].to->value += tanh(links[i].from->value) * links[i].weight;
        }
    }

    ///get active output node
    double max = 0;
    int maxID = -1;
    for(int i = 0; i < NODE_COUNT; i++){
        if(nodes[i].type == OUTPUT && nodes[i].value > max){
            max = nodes[i].value;
            maxID = i;
        }
    }
    if(maxID == -1) return;

    ///perform action of active output node
    switch('A' + maxID - INPUT_NODES - HIDDEN_NODES){
        case 'A':{ //stay
            break;
        }
        case 'B':{ //move x+
            changePosition(1, 0);
            forwardDir = 0;
            break;
        }
        case 'C':{ //move x-
            changePosition(-1, 0);
            forwardDir = 2;
            break;
        }
        case 'D':{ //move y+
            changePosition(0, 1);
            forwardDir = 3;
            break;
        }
        case 'E':{ //move y-
            changePosition(0, -1);
            forwardDir = 1;
            break;
        }
        case 'F':{ //move random
            int r = randInt(randomGen) % 4;
            changePosition(r == 0 ? 1 : r == 2 ? -1 : 0, r == 3 ? 1 : r == 1 ? -1 : 0);
            forwardDir = r;
            break;
        }
        case 'G':{ //move forward
            changePosition(forwardDir == 0 ? 1 : forwardDir == 2 ? -1 : 0, forwardDir == 3 ? 1 : forwardDir == 1 ? -1 : 0);
            break;
        }
        case 'H':{ //turn left
            forwardDir = (forwardDir + 1) % 4;
            break;
        }
        case 'I':{ //turn right
            forwardDir = (forwardDir + 3) % 4;
            break;
        }
        case 'J':{ //turn around
            forwardDir = (forwardDir + 2) % 4;
            break;
        }
        default:{
            printf("Houston we have a problem!\n");
        }
    }
}

