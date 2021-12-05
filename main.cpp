#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma ide diagnostic ignored "modernize-loop-convert"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <windows.h>
#include "neuralnet.h"

HANDLE hConsole;
int survived = 0;
int recordGen = 1000;
std::string** survivedDna;
int*** imageData;
int** occupiedPositions;

int surviveCondition(const Neuralnet& n);
void colorPrint(const char* message, int color);
void createImage(int frame);

int main(){
    ///malloc
    occupiedPositions = new int*[BOARD_SIZE];
    for(int i = 0; i < BOARD_SIZE; ++i){
        occupiedPositions[i] = new int[BOARD_SIZE];
    }

    survivedDna = new std::string*[POPULATION];
    for(int i = 0; i < POPULATION; i++){
        survivedDna[i] = new std::string[DNA_SEGMENT_COUNT];
    }

    imageData = new int**[BOARD_SIZE];
    for(int i = 0; i < BOARD_SIZE; i++){
        imageData[i] = new int*[BOARD_SIZE];
        for(int j = 0; j < BOARD_SIZE; j++){
            imageData[i][j] = new int[3];
        }
    }

    ///create folder if doesn't exist
    system("IF NOT exist ..\\images mkdir ..\\images");

    ///text color
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
        ///reset
        for(int k = 0; k < BOARD_SIZE; k++){
            for(int l = 0; l < BOARD_SIZE; l++){
                occupiedPositions[k][l] = 0;
            }
        }

        ///main loop
        for(int p = 0; p < POPULATION; p++){
            if(i == 0){
                if(load == 'n') n[p].reset(nullptr);
                else{
                    std::string loadedDna[DNA_SEGMENT_COUNT];
                    for(int j = 0; j < DNA_SEGMENT_COUNT; j++){
                        char temp[DNA_SEGMENT_SIZE];
                        fscanf(log_read, "%s", temp);
                        loadedDna[j].append(temp);
                    }
                    n[p].reset(loadedDna);
                }
            }
            else{
                if(p < survived) n[p].reset(survivedDna[p]);
                else n[p].reset(survivedDna[randInt(randomGen) % survived]);
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
    }
    fclose(log_write);

    ///free
    delete[] occupiedPositions;
    delete[] survivedDna;
    delete[] imageData;
    for(int i = 0; i < POPULATION; i++){
        delete[] n[i].links;
        delete[] n[i].nodes;
        delete[] n[i].dna;
    }

    colorPrint("Press ENTER to close window...", 15);
    getchar();
    return 0;
}

int surviveCondition(const Neuralnet& n){
    double r = sqrt(pow(n.pos.x - 50, 2) + pow(n.pos.y - 50, 2));
    double dev = abs(r - 20);
    if(randInt(randomGen) % 150 > 100 * dev - 50) return 1;
    else return 0;
}
void colorPrint(const char* message, int color){
    SetConsoleTextAttribute(hConsole, color);
    printf("%s", message);
    SetConsoleTextAttribute(hConsole, 15);
}
void createImage(int frame){
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
