#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define DEBUG 1
#define FALSE 0
#define TRUE 1

#define LEFT_GOAL -1
#define RIGHT_GOAL 1
#define NO_GOAL 0

#define NO_WINNER -1

#define WIDTH 96
#define LENGTH 128
#define NUM_ROUNDS 200

typedef struct
{
    int x;
    int y;
} pos;

typedef struct
{
    int id;         // player id
    pos initial;    // initial pos 
    pos final;      // final pos

    int reached;
    int kicked;
    int challenge;

    // attributes
    int speed;
    int dribbling;
    int kick;
} football_player;

int field, tag;

void initField(int world_rank, int* goalA, int* goalB, pos* ball);
void initPlayers(int world_rank, football_player* player);

// identifiers
int isFieldProcess(int worldRank);
int isPlayerProcess(int worldRank);
int isTeamA(int worldRank);
int isTeamB(int worldRank);
int isFP0(int worldRank);
int isGoal(pos ball);
int isBallWithinRange(pos ball, football_player player);

// helper functions
int getFieldProcess(pos player);
void getRandomPos(int worldRank, pos* target);
void tryToReach(pos target, football_player* player);
void incrementScore(pos ball, int goalA, int goalB, int* Ascore, int* Bscore);
void groupFieldAndPlayers(int worldRank, football_player player, int* field, MPI_Comm* subfield_comm);
void groupAllFieldProcesses(int worldRank, MPI_Comm* field_comm);
void groupFP0AndPlayers(int worldRank, MPI_Comm* reporting_comm);
void swapGoals(int* goalA, int* goalB);
void challengeBall(int worldRank, int challenge[2]);
void aimBall(pos* target, football_player player, int goal);
void kickBall(pos target, pos* ball);
void moveTo(pos target, football_player* player);
void handleFieldWithBall(int worldRank, football_player* player, pos* ball, MPI_Comm subfield_comm, MPI_Datatype mpi_ball, int goalA, int goalB);

// print functions
void printPlayerInfo(football_player players[23]);
void printFieldGroups(int worldRank, int worldSize, MPI_Comm subfield_comm);

// facades
void startHalf(int worldRank, football_player* player);
void startRound(football_player* player);

void createBallStruct(MPI_Datatype* mpi_ball);
void createPlayerStruct(MPI_Datatype mpi_ball, MPI_Datatype* mpi_player);

int main(int argc, char **argv)
{
    int worldSize, worldRank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int round, half;
    int field, winner;
    int playerChangeField = 0;
    int goalA, goalB;
    int Ascore = 0;
    int Bscore = 0;

    srand(worldRank);
    
    pos ball;
    MPI_Datatype mpi_ball;
    createBallStruct(&mpi_ball);

    football_player player;
    MPI_Datatype mpi_player;
    createPlayerStruct(mpi_ball, &mpi_player);

    player.id = worldRank;
    initField(worldRank, &goalA, &goalB, &ball);
    initPlayers(worldRank, &player);

    MPI_Comm field_comm, reporting_comm, subfield_comm;
    groupAllFieldProcesses(worldRank, &field_comm);
    groupFP0AndPlayers(worldRank, &reporting_comm);
    
    for (half = 0; half < 2; half++) {
        swapGoals(&goalA, &goalB);
        startHalf(worldRank, &player);
        groupFieldAndPlayers(worldRank, player, &field, &subfield_comm);
        for (round = 0; round < NUM_ROUNDS; round++) {
            startRound(&player);
            // get ball position from field process
            MPI_Bcast(&ball, 1, mpi_ball, 0, subfield_comm);
            int fieldWithBall = getFieldProcess(ball);

            playerChangeField = FALSE;
            if (isPlayerProcess(worldRank))
            {
                
               if (field == fieldWithBall)
               {
                   // run after ball
                   tryToReach(ball, &player);
               }
               else if (isBallWithinRange(ball, player)) 
               {
                   // check if ball is within range
                   moveTo(ball, &player);
               } 
               else {
                   // move to default position
                   pos target;
                   getRandomPos(worldRank, &target);
                   tryToReach(target, &player);
               }
               // check if field changed
               if (field != getFieldProcess(player.final))
               {
                    playerChangeField = TRUE;
               }
            }

            int p;
            int playersFieldChange[34];
            MPI_Allgather(&playerChangeField, 1, MPI_INT, &playersFieldChange, 1, MPI_INT, MPI_COMM_WORLD);
            for (p = 0; p < 34; p++) 
            {
                if (playersFieldChange[p] != FALSE) 
                {
                    playerChangeField = TRUE;
                }
            }
            // comm now outdated; regroup into new comm
            if (playerChangeField) {
                MPI_Comm_free(&subfield_comm);
                groupFieldAndPlayers(worldRank, player, &field, &subfield_comm);
            }

            // Field with ball will handle ball challenges
            if (field == fieldWithBall)
            {
                handleFieldWithBall(worldRank, &player, &ball, subfield_comm, mpi_ball, goalA, goalB);
            }

            // field with ball broadcasts new ball position to all fields
            if (isFieldProcess(worldRank))
            {
                MPI_Bcast(&ball, 1, mpi_ball, fieldWithBall, field_comm);
                if (isGoal(ball))
                {
                    // increment score
                    incrementScore(ball, goalA, goalB, &Ascore, &Bscore);
                    // reset ball position
                    ball.x = WIDTH / 2;
                    ball.y = LENGTH / 2;
                }
            }

            // all players send their position to field 0 
            if (isFP0(worldRank) || isPlayerProcess(worldRank))
            {
                football_player players[23];
                MPI_Gather(&player, 1, mpi_player, &players, 1, mpi_player, 0, reporting_comm);
                if (isFP0(worldRank)) 
                {
                    printf("Round: %d\n", round);
                    printf("Ball: %d %d\n", ball.x, ball.y);
                    // printPlayerInfo(players);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    if (isFP0(worldRank)) printf("Final score: A %d:%d B\n", Ascore, Bscore);
    
    MPI_Comm_free(&subfield_comm);
    MPI_Comm_free(&reporting_comm);
    MPI_Comm_free(&field_comm);
    MPI_Finalize();
}

void initField(int worldRank, int* goalA, int* goalB, pos* ball)
{
    // Goals will be swapped after this (i.e. goalA = LEFT, goalB = RIGHT)
    *goalA = RIGHT_GOAL;
    *goalB = LEFT_GOAL;

    if (isFieldProcess(worldRank)) 
    {
        // Ball starts in the center
        ball->x = LENGTH/2;
        ball->y = WIDTH/2;
    }
}

void initPlayers(int worldRank, football_player* player)
{
    // player stats
    pos target;
    getRandomPos(worldRank, &target);
    player->id = worldRank;
    player->initial.x = target.x;
    player->initial.y = target.y;
    player->final.x = player->initial.x;
    player->final.y = player->initial.y;
    player->reached = 0;
    player->kicked = 0;
    player->challenge = -1;
    player->speed = 10;
    player->dribbling = 1;
    player->kick = 4;
    
}

void getRandomPos(int worldRank, pos* target) 
{
    int field = worldRank % 12;
    target->x = field % 4 * 32 + rand() % 32;
    target->y = field / 4 * 32 + rand() % 32;
}

void swapGoals(int* goalA, int* goalB)
{
    *goalA = (*goalA == LEFT_GOAL) ?  RIGHT_GOAL : LEFT_GOAL;
    *goalB = (*goalB == LEFT_GOAL) ?  RIGHT_GOAL : LEFT_GOAL;
}

void incrementScore(pos ball, int goalA, int goalB, int* Ascore, int* Bscore)
{
    int goal;
    if (ball.x == 0) goal = LEFT_GOAL;
    else if (ball.x == 127) goal = RIGHT_GOAL;

    if (goal == goalA) {
        (*Ascore)++;
    }
    else if (goal == goalB) {
        (*Bscore)++;
    }
}

int isFieldProcess(int worldRank)
{
    return (worldRank < 12);
}

int isPlayerProcess(int worldRank)
{
    return (worldRank >= 12 && worldRank < 34);
}

int isTeamA(int worldRank)
{
    return (worldRank >= 12 && worldRank < 23);
}

int isTeamB(int worldRank)
{
    return (worldRank >= 23 && worldRank < 34);
}

int isFP0(int worldRank) 
{
    return worldRank == 0;
}

int isGoal(pos ball)
{
    if (ball.x == 0 && ball.y >= 43 && ball.y <= 51) return LEFT_GOAL;
    else if (ball.x == 127 && ball.y >= 43 && ball.y <= 51) return RIGHT_GOAL;
    else return NO_GOAL;
}

int isBallWithinRange(pos ball, football_player player) 
{
    int moves_left = (player.speed < 10 ? player.speed : 10);
    
    int diff = player.initial.x - ball.x;
    if (player.initial.x < ball.x) diff *= -1;
    if (diff < moves_left) moves_left -= diff;
    else return 0;

    diff = player.initial.y - ball.y;
    if (player.initial.y < ball.y) diff *= -1;
    if (diff < moves_left)  moves_left -= diff;
    else return 0;

    return (moves_left >= 0);
}

void groupFieldAndPlayers(int worldRank, football_player player, int* field, MPI_Comm* subfield_comm) 
{
    if (isFieldProcess(worldRank))
    {
        *field = worldRank;
    } 
    else if (isPlayerProcess(worldRank)) 
    {
        
        *field = getFieldProcess(player.final);
    }
    MPI_Comm_split(MPI_COMM_WORLD, *field, worldRank, subfield_comm);
}

void groupAllFieldProcesses(int worldRank, MPI_Comm* field_comm)
{
    int colour = 1;
    if (isFieldProcess(worldRank)) colour = 0;
    MPI_Comm_split(MPI_COMM_WORLD, colour, worldRank, field_comm);
}

void groupFP0AndPlayers(int worldRank, MPI_Comm* reporting_comm) 
{
    int colour = 1;
    if (isFP0(worldRank) || isPlayerProcess(worldRank)) colour = 0;
    MPI_Comm_split(MPI_COMM_WORLD, colour, worldRank, reporting_comm);
}

int getFieldProcess(pos player) 
{
    if (player.y >= 0 && player.y <= 31)
    {
        if (player.x >= 0 && player.x <= 31) return 0;
        if (player.x >= 32 && player.x <= 63) return 1;
        if (player.x >= 64 && player.x <= 95) return 2;
        if (player.x >= 95 && player.x <= 127) return 3;
    }
    else if (player.y >= 32 && player.y <= 63)
    {
        if (player.x >= 0 && player.x <= 31) return 4;
        if (player.x >= 32 && player.x <= 63) return 5;
        if (player.x >= 64 && player.x <= 95) return 6;
        if (player.x >= 95 && player.x <= 127) return 7;
    }
    else if (player.y >= 64 && player.y <= 95)
    {
        if (player.x >= 0 && player.x <= 31) return 8;
        if (player.x >= 32 && player.x <= 63) return 9;
        if (player.x >= 64 && player.x <= 95) return 10;
        if (player.x >= 95 && player.x <= 127) return 11;
    }
    printf("Error: invalid player %d %d\n", player.x, player.y);
    return -1;
}

void tryToReach(pos target, football_player* player)
{
    player->final.x = player->initial.x;
    player->final.y = player->initial.y;
    int moves_left = (player->speed < 10 ? player->speed : 10);

    int diff = player->initial.x - target.x;
    if (player->initial.x > target.x) 
    {
        int move = (diff < moves_left ? diff : moves_left);
        player->final.x = player->initial.x - move;
    }
    diff = target.x - player->initial.x;
    if (player->initial.x < target.x) 
    {
        int move = (diff < moves_left ? diff : moves_left);
        player->final.x = player->initial.x + move;
    }

    diff = player->initial.y - target.y;
    if (player->initial.y > target.y) 
    {
        int move = (diff < moves_left ? diff : moves_left);
        player->final.y = player->initial.y - move;
    }
    diff = target.y - player->initial.y;
    if (player->initial.y < target.y) 
    {
        int move = (diff < moves_left ? diff : moves_left);
        player->final.y = player->initial.y + move;
    }
}

void aimBall(pos* target, football_player player, int goal)
{
    int moves_left = player.kick * 2;
    target->x = player.final.x;
    target->y = player.final.y;
    int direction = rand() % 2;
    if (goal == LEFT_GOAL) {
        // move in x first
        if (player.final.x != 0) {
            int diff = player.final.x;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->x = player.final.x - move;
        }
        if (player.final.y < 43)
        {
            int diff = 43 - player.final.y;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->y = player.final.y + move;
        }
        if (player.final.y > 51)
        {
            int diff = player.final.y - 51;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->y = player.final.y - move;
        }
    } 
    else 
    {
        // move in x first
        if (player.final.x != 127) {
            int diff = 127 - player.final.x;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->x = player.final.x + move;
        }
        if (player.final.y < 43)
        {
            int diff = 43 - player.final.y;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->y = player.final.y + move;
        }
        if (player.final.y > 51)
        {
            int diff = player.final.y - 51;
            int move = (diff < moves_left ? diff : moves_left);
            moves_left -= move;
            target->y = player.final.y - move;
        }
    }
    // printf("target %d %d\n", target->x, target->y);
}

void kickBall(pos target, pos* ball) 
{
    ball->x = target.x;
    ball->y = target.y;
}

void moveTo(pos target, football_player* player)
{
    player->final.x = target.x;
    player->final.y = target.y;
}

void printPlayerInfo(football_player player[23]) 
{
    int p;
    for (p = 0; p < 23; p++)
    {
        if (player[p].id > 11 && player[p].id < 23)
        {
            printf("id: %d; ", player[p].id - 12);
        }
        if (player[p].id > 22 && player[p].id < 34)
        {
            printf("id: %d; ", player[p].id - 23);
        }
        if (player[p].id > 11) 
        {
            printf("initial: %d, %d; ", player[p].initial.x, player[p].initial.y);
            printf("final: %d %d; ", player[p].final.x, player[p].final.y);
            printf("reached: %d; ", player[p].reached);
            printf("kicked: %d; ", player[p].kicked);
            printf("challenge: %d; ", player[p].challenge);
            printf("\n");
        }
    }
}

void printFieldGroups(int worldRank, int worldSize, MPI_Comm subfield_comm)
{
    int fieldRank, fieldSize;
    MPI_Comm_rank(subfield_comm, &fieldRank);
    MPI_Comm_size(subfield_comm, &fieldSize);
    printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d\n", worldRank, worldSize, fieldRank, fieldSize);
}

void startHalf(int worldRank, football_player* player) 
{
    player->id = worldRank;
    player->initial.x = player->final.x;
    player->initial.y = player->final.y;
    player->kicked = 0;
    player->reached = 0;
    player->challenge = -1;
    
}

void startRound(football_player* player)
{
    player->initial.x = player->final.x;
    player->initial.y = player->final.y;
    player->kicked = 0;
    player->reached = 0;
    player->challenge = -1;
}

void createBallStruct(MPI_Datatype* mpi_ball) 
{
    int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(pos, x);
    offsets[1] = offsetof(pos, y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_ball);
    MPI_Type_commit(mpi_ball);
}

void createPlayerStruct(MPI_Datatype mpi_ball, MPI_Datatype* mpi_player) 
{
    const int nitems = 9;
    int blocklengths[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[9] = {MPI_INT, mpi_ball, mpi_ball, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint     offsets[9];

    offsets[0] = offsetof(football_player, id);
    offsets[1] = offsetof(football_player, initial);
    offsets[2] = offsetof(football_player, final);
    offsets[3] = offsetof(football_player, reached);
    offsets[4] = offsetof(football_player, kicked);
    offsets[5] = offsetof(football_player, challenge);
    offsets[6] = offsetof(football_player, speed);
    offsets[7] = offsetof(football_player, dribbling);
    offsets[8] = offsetof(football_player, kick);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_player);
    MPI_Type_commit(mpi_player);
}

void handleFieldWithBall(int worldRank, football_player* player, pos* ball, MPI_Comm subfield_comm, MPI_Datatype mpi_ball, int goalA, int goalB) 
{
    
    int fieldSize, fieldRank;
    MPI_Comm_rank(subfield_comm, &fieldRank);
    MPI_Comm_size(subfield_comm, &fieldSize);

    int challenge[2], winner;
    challenge[0] = 0;
    challenge[1] = fieldRank;
    if (isPlayerProcess(worldRank) && player->final.x == ball->x && player->final.y == ball->y) 
    {
        challenge[0] = (rand() % 9) + 1;
        challenge[0] *= player->dribbling;
        player->challenge = challenge[0];
        player->reached = 1;
    }
    
    int challenges[44];
    MPI_Gather(&challenge, 2, MPI_INT, &challenges, 2, MPI_INT, 0, subfield_comm);

    if (isFieldProcess(worldRank))
    {
        int p, topChallenge;
        topChallenge = 0;
        winner = NO_WINNER;
        for (p = 0; p < 2*fieldSize; p+=2)
        {
            // printf("(%d,%d) ", challenges[p+1], challenges[p]);
            if (challenges[p+1] != 0 && challenges[p] > topChallenge)
            {
                topChallenge = challenges[p];
                winner = challenges[p+1];
            }
        }
        // printf("\n");
    }
    
    // broadcast winner to all participants
    MPI_Bcast(&winner, 1, MPI_INT, 0, subfield_comm);
    
    if (winner != NO_WINNER)
    {
        // Winning player kicks the ball
        if (fieldRank == winner) {
            // printf("winner: %d\n", worldRank);

            pos target;
            if (isTeamA(worldRank)) aimBall(&target, *player, goalA);
            else if (isTeamB(worldRank)) aimBall(&target, *player, goalB);
            kickBall(target, ball);
            player->kicked = 1;
            // printf("kicked to %d %d\n", ball->x, ball->y);

        }
        // broadcast new ball position to field
        MPI_Bcast(ball, 1, mpi_ball, winner, subfield_comm);

    }
}