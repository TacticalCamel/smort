#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma ide diagnostic ignored "modernize-loop-convert"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <windows.h>

///general simulation settings
const int CPU_THREADS = 12;
const int DNA_SEGMENT_SIZE = 6;
const int BOARD_SIZE = 100;
const int INPUT_NODES = 10;
const int OUTPUT_NODES = 10;

const int HIDDEN_NODES = 8;
const int NODE_COUNT = INPUT_NODES + OUTPUT_NODES + HIDDEN_NODES;

const int GENERATIONS = 1000;
const int POPULATION = 500;
const int SIMULATION_STEPS = 200;
const int DNA_SEGMENT_COUNT = 25;
const double MUTATION_CHANCE = 0.002;

///global variables
HANDLE hConsole;
std::default_random_engine randomGen(time(nullptr));
std::uniform_int_distribution<int> randInt(0, INT_MAX);
std::string survivedDna[POPULATION][DNA_SEGMENT_COUNT];
int occupiedPositions[BOARD_SIZE][BOARD_SIZE];
int imageData[BOARD_SIZE][BOARD_SIZE][3];
int survived = 0;
int recordGen = 1000;

class Neuralnet{
private:
    ///private variables
    typedef enum {INPUT, HIDDEN, OUTPUT} NodeType;
    typedef struct{
        int x;
        int y;
    } Vector2;
    typedef struct{
        NodeType type;
        double value;
    } Node;
    typedef struct{
        double weight;
        Node* from;
        Node* to;
    } Link;

    ///internal functions
    void changePosition(int x, int y){
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
    void setDna(std::string* inheritedDna){
        //random dna
        if(inheritedDna == nullptr){
            for(int i = 0; i < DNA_SEGMENT_COUNT; i++){
                dna[i].clear();
                for(int j = 0; j < DNA_SEGMENT_SIZE; j++){
                    int x = (j == 0) ? randInt(randomGen) % (INPUT_NODES + HIDDEN_NODES) : (j == 1) ? randInt(randomGen) % (HIDDEN_NODES + OUTPUT_NODES) + INPUT_NODES : randInt(randomGen) % NODE_COUNT;
                    dna[i].append(1, x < 10 ? (char)('0' + x) : (char)('A' + x - 10));
                }
            }
        }
        //inherited dna
        else{
            for(int i = 0; i < DNA_SEGMENT_COUNT; i++){
                dna[i] = inheritedDna[i];
                for(int j = 0; j < DNA_SEGMENT_SIZE; j++){
                    if((double)randInt(randomGen) / INT_MAX < MUTATION_CHANCE){
                        int x = (j == 0) ? randInt(randomGen) % (INPUT_NODES + HIDDEN_NODES) : (j == 1) ? randInt(randomGen) % (HIDDEN_NODES + OUTPUT_NODES) + INPUT_NODES : randInt(randomGen) % NODE_COUNT;
                        dna[i][j] = x < 10 ? (char)('0' + x) : (char)('A' + x - 10);
                    }
                }
            }
        }
    }
    void setRandomPosition(){
        do{
            pos.x = randInt(randomGen) % BOARD_SIZE;
            pos.y = randInt(randomGen) % BOARD_SIZE;
        } while(occupiedPositions[pos.x][pos.y] == 1);
        occupiedPositions[pos.x][pos.y] = 1;
    }
    void createNodes(){
        for(int i = 0; i < NODE_COUNT; i++){
            nodes[i].type = i < INPUT_NODES ? INPUT : (i < INPUT_NODES + HIDDEN_NODES ? HIDDEN : OUTPUT);
        }
    }
    void createLinks(){
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

public:
    ///public properties
    Vector2 pos;
    int age;
    int linkCount;
    int forwardDir;
    Node nodes[NODE_COUNT];
    std::string dna[DNA_SEGMENT_COUNT];
    Link links[DNA_SEGMENT_COUNT];

    ///set starting properties
    void initialize(std::string* inheritedDna){
        age = 0;
        linkCount = 0;
        forwardDir = randInt(randomGen) % 4;

        setDna(inheritedDna);
        setRandomPosition();
        createNodes();
        createLinks();
    }

    ///process 1 simulation step
    void step(){
        for(int i = 0; i < NODE_COUNT; i++){
            nodes[i].value = 0;
        }
        ///set input node values: [0; 1]
        //const 1
        nodes[0].value = 1;
        //x position
        nodes[1].value = (double)pos.x / BOARD_SIZE;
        //y position
        nodes[2].value = (double)pos.y / BOARD_SIZE;
        //age
        nodes[3].value = (double)(age++) / SIMULATION_STEPS;
        //random
        nodes[4].value = (double)randInt(randomGen) / INT_MAX;
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
                links[i].to->value += tanh(links[i].from->value) * links[i].weight;
            }
        }
        //hidden -> output
        for(int i = 0; i < linkCount; i++){
            if(links[i].from->type == HIDDEN){
                links[i].to->value += tanh(links[i].from->value) * links[i].weight;
            }
        }
        //normalize output
        for(int i = 0; i < linkCount; i++){
            if(links[i].to->type == OUTPUT){
                links[i].to->value = tanh(links[i].to->value);
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
};

int surviveCondition(const Neuralnet& n);
void colorPrint(const char* message, int color);
void createImage(int frame);

int main(){
    //text color
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);

    Neuralnet n[POPULATION];
    FILE* log_read = fopen("..\\log.txt", "r");

    char load;
    do{
        colorPrint("Load from file? ", 14);
        fflush(stdin);
        load = getchar();
    } while(load != 'y' && load != 'n');

    if(load == 'y') colorPrint("Loading file...\n",10);
    else colorPrint("Generating random instances...\n", 4);

    for(int i = 0; i < GENERATIONS; i++){
        ///initialize
        for(int k = 0; k < BOARD_SIZE; k++){
            for(int l = 0; l < BOARD_SIZE; l++){
                occupiedPositions[k][l] = 0;
            }
        }

        ///repopulate
        for(int p = 0; p < POPULATION; p++){
            if(i == 0){
                if(load == 'n') n[p].initialize(nullptr);
                else{

                    std::string loadedDna[DNA_SEGMENT_COUNT];
                    for(int j = 0; j < DNA_SEGMENT_COUNT; j++){
                        char temp[DNA_SEGMENT_SIZE];
                        fscanf(log_read, "%s", temp);
                        loadedDna[j].append(temp);
                    }

                    ///DEBUG FUUUU

                    n[p].initialize(loadedDna);

                }
            }
            else{
                //nem a legjobb módszer, de egyelőre jó lesz
                if(p < survived) n[p].initialize(survivedDna[p]);
                else n[p].initialize(survivedDna[randInt(randomGen) % survived]);
            }
        }

        ///process simulation steps
        for(int j = 0; j < SIMULATION_STEPS; j++){
            //empty image
            if(i % recordGen == recordGen - 1){
                for(int y = 0; y < BOARD_SIZE; y++){
                    for(int x = 0; x < BOARD_SIZE; x++){
                        imageData[x][y][0] = 0;
                        imageData[x][y][1] = 0;
                        imageData[x][y][2] = 0;
                    }
                }
            }

            //steps
            #pragma omp parallel for schedule(dynamic), num_threads(CPU_THREADS)
            for(int p = 0; p < POPULATION; p++){
                n[p].step();
                if(i % recordGen == recordGen - 1){
                    imageData[n[p].pos.x][n[p].pos.y][0] = randInt(randomGen) % 256;
                    imageData[n[p].pos.x][n[p].pos.y][1] = randInt(randomGen) % 256;
                    imageData[n[p].pos.x][n[p].pos.y][2] = randInt(randomGen) % 256;
                }
            }

            //create image
            if(i % recordGen == recordGen - 1){
                createImage(j);
            }
        }

        ///decide if survived or not
        survived = 0;
        for(int p = 0; p < POPULATION; p++){
            if(surviveCondition(n[p])){
                for(int k = 0; k < DNA_SEGMENT_COUNT; k++){
                    survivedDna[survived][k] = n[p].dna[k];
                }
                survived++;
            }
        }

        ///console
        if(i % 5 == 0) printf("\rgen#%05d | survived: %03d", i, survived);
        if(i % 100 == 0){
            int selected = randInt(randomGen) % survived;
            for(int j = 0; j < DNA_SEGMENT_COUNT; j++){
                int usingSegment = 1;

                //1. source cannot be OUTPUT type
                if(survivedDna[selected][j][0] >= 'A' + HIDDEN_NODES) usingSegment = 0;
                //2. destination cannot be INPUT type
                if(survivedDna[selected][j][1] < 'A') usingSegment = 0;
                //3. no duplicate connections
                for(int k = 0; k < j; k++){
                    if(survivedDna[selected][k][0] == survivedDna[selected][j][0] && survivedDna[selected][k][1] == survivedDna[selected][j][1]) usingSegment = 0;
                }
                SetConsoleTextAttribute(hConsole, usingSegment ? 11 : 8);
                printf(" %s", survivedDna[selected][j].data());
                SetConsoleTextAttribute(hConsole, 15);
            }
            printf("\n");
        }
        ///
    }
    fclose(log_read);
    FILE* log_write = fopen("..\\log.txt", "r+");

    SetConsoleTextAttribute(hConsole, 10);
    printf("\n\nProcess finished with exit code 0\n\a");
    char save;
    do{
        colorPrint("Overwrite save file? ", 14);
        fflush(stdin);
        save = getchar();
    } while(save != 'y' && save != 'n');

    if(save == 'n'){
        colorPrint("Discarded!\n", 4);
        fflush(stdin);
        getchar();
    }
    else{
        fclose(fopen("..\\log.txt", "w"));
        for(int i = 0; i < POPULATION; i++){
            for(int j = 0; j < DNA_SEGMENT_COUNT; j++){
                fprintf(log_write, "%s%c", n[i].dna[j].data(), j == DNA_SEGMENT_COUNT - 1 ? '\n' : ' ');
            }
        }

        colorPrint("Saving complete!\n", 10);
        fflush(stdin);
        getchar();
    }
    fclose(log_write);

    return 0;
}

int surviveCondition(const Neuralnet& n){
    double r = sqrt(pow(n.pos.x - 50, 2) + pow(n.pos.y - 50, 2));
    if(r > 38 && r < 40) return 1;
    return 0;
}
void colorPrint(const char* message, int color){
    SetConsoleTextAttribute(hConsole, color);
    printf("%s", message);
    SetConsoleTextAttribute(hConsole, 15);
}
void createImage(int frame){
    system("IF NOT exist ..\\images mkdir ..\\images");
    std::string name = "..\\images\\image_";
    name.append(1, (char)('0' + (frame / 100) % 10));
    name.append(1, (char)('0' + (frame / 10) % 10));
    name.append(1, (char)('0' + frame % 10));
    name.append(".pgm");
    FILE* imageFile = fopen(name.data(), "w");
    if(imageFile == nullptr){
        colorPrint("\nImageFile %s is nullpointer!\n", 4);
        return;
    }
    fprintf(imageFile, "P3\n%d %d\n255\n", BOARD_SIZE, BOARD_SIZE);
    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            fprintf(imageFile, "%d %d %d\n", imageData[j][i][0], imageData[j][i][1], imageData[j][i][2]);
        }
    }
    fclose(imageFile);
}
