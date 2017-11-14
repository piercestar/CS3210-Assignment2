#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define WIDTH 64
#define LENGTH 128
#define NUM_ROUNDS 20
#define COUNT_1 1
#define DEBUG 1

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
    int ran;        // distance ran
    int reached;    // no. of times reached the ball
    int kicked;     // no. of times kicked the ball
} football_player;

int field, tag;

void initialize(pos* ball, football_player* player, int rank) 
{
    if (rank == field) 
    {
        // ball starts in the middle
        ball->x = LENGTH / 2 ;
        ball->y = WIDTH / 2 ;
    }
    else // initialize player
    {
        player->id = rank;
        player->initial.x = rand() % LENGTH;
        player->initial.y = rand() % WIDTH;
        player->final.x = player->initial.x;
        player->final.y = player->initial.y;
        player->ran = 0;
        player->reached = 0;
        player->kicked = 0;
    }
}

void createBallStruct(MPI_Datatype* mpi_ball) {
    int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(pos, x);
    offsets[1] = offsetof(pos, y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_ball);
    MPI_Type_commit(mpi_ball);
}

void createPlayerStruct(MPI_Datatype mpi_ball, MPI_Datatype* mpi_player) {
    const int nitems = 6;
    int blocklengths[6] = {1, 1, 1, 1, 1, 1};
    MPI_Datatype types[6] = {MPI_INT, mpi_ball, mpi_ball, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint     offsets[6];

    offsets[0] = offsetof(football_player, id);
    offsets[1] = offsetof(football_player, initial);
    offsets[2] = offsetof(football_player, final);
    offsets[3] = offsetof(football_player, ran);
    offsets[4] = offsetof(football_player, reached);
    offsets[5] = offsetof(football_player, kicked);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_player);
    MPI_Type_commit(mpi_player);
}

int has_reached(int id, int reached[], int numReached) {
    int i;
    for (i = 0; i < numReached; i++)
    {
        if (reached[i] == id) return 1;
    }
    return 0;
}

void print_player_data(football_player player, int has_reached, int has_kicked) {
    printf("Player %d; ", player.id);
    printf("init_pos: %d, %d; ", player.initial.x, player.initial.y);
    printf("final_pos: %d, %d; ", player.final.x, player.final.y);
    printf("has_reached: %d; ", has_reached);
    printf("has_kicked: %d; ", has_kicked);
    printf("total_ran: %d; ", player.ran);
    printf("total_reached: %d; ", player.reached);
    printf("total_kicked: %d; ", player.kicked);
    printf("\n");
}

void move_player(football_player* player, pos ball) {
    int moves_left = 10;
    int movement;
    int direction;
    int difference;
    if (player->initial.x != ball.x && moves_left != 0) {
        if (player->initial.x > ball.x) {
            direction = -1;
            difference = player->initial.x - ball.x;
        } else {
            direction = 1;
            difference = ball.x - player->initial.x;
        }
        movement = difference < moves_left ? difference : moves_left;
        player->final.x = player->initial.x + direction * movement;
        moves_left -= movement;
    }

    if (player->initial.y != ball.y && moves_left != 0) {
        if (player->initial.y > ball.y) {
            direction = -1;
            difference = player->initial.y - ball.y;
        } else {
            direction = 1;
            difference = ball.y - player->initial.y;
        }  
        movement = difference < moves_left ? difference : moves_left;
        player->final.y = player->initial.y + direction * movement;
        moves_left -= movement;
    }

    // increment dist ran
    player->ran += 10 - moves_left;
    // increment reached
    if (ball.x == player->final.x && ball.y == player->final.y)
        player->reached += 1;
}

void determine_kicker(int* kicker, int reached[], int* numReached, int num_p, football_player players[], pos ball) 
{
    int i;
    for (i = 0; i < num_p; i++)
    {
        // find out all the players that reached the ball
        if (players[i].final.x == ball.x && players[i].final.y == ball.y) 
        {
            reached[*numReached] = i;
            (*numReached)++;
        }
    }

    if (*numReached != 0) 
    {
        // randomly select the kicker
        *kicker = reached[rand() % *numReached];
    } 
    else 
    {
        // no kicker
        *kicker = -1;
    }

}

int main(int argc, char **argv)
{
    int world_size, world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int round, p, kicker;
    int num_p = world_size - 1;

    field = world_size - 1;
    tag = 0;

    // create a type for struct ball
    pos ball;
    MPI_Datatype mpi_ball;
    createBallStruct(&mpi_ball);

    // create a type for struct players
    football_player player;
    MPI_Datatype mpi_player;
    createPlayerStruct(mpi_ball, &mpi_player);

    srand(world_rank);

    initialize(&ball, &player, world_rank);

    for (round = 0; round < NUM_ROUNDS; round++) {
        // field process
        if (world_rank == field)
        {
            football_player players[num_p];     // array for storing player data

            // start new round
            printf("Round: %d\n", round);
            printf("Ball: %d, %d\n", ball.x, ball.y);

            // Send out new ball position
            for (p = 0; p < num_p; p++) 
            {
                MPI_Send(&ball, COUNT_1, mpi_ball, p, tag, MPI_COMM_WORLD);
            }

            // Receive final positions
            for (p = 0; p < num_p; p++)
            {
                MPI_Recv(&players[p], COUNT_1, mpi_player, p, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // determine kicker
            kicker = -1;
            int numReached = 0;
            int reached[11];
            determine_kicker(&kicker, reached, &numReached, num_p, players, ball);
            if (DEBUG) printf("%d players reached\n", numReached);

            // announce kicker
            for (p = 0; p < num_p; p++) 
            {
                MPI_Send(&kicker, COUNT_1, MPI_INT, p, tag, MPI_COMM_WORLD);
            }

            // Update kicker information
            if (numReached > 0) {
                MPI_Recv(&ball, COUNT_1, mpi_ball, kicker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&players[kicker], COUNT_1, mpi_player, kicker, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } 

            // Output player results
            for (p = 0; p < num_p; p++) {
                print_player_data(players[p], has_reached(p, reached, numReached), kicker == p ? 1 : 0);
            }
        }
        else // player process
        {
            // set new initial positions
            player.initial.x = player.final.x;
            player.initial.y = player.final.y;

            // receive ball location
            MPI_Recv(&ball, COUNT_1, mpi_ball, field, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // move towards the ball
            move_player(&player, ball);
            
            // Send final positions
            MPI_Send(&player, COUNT_1, mpi_player, field, tag, MPI_COMM_WORLD); 

            // Get kicker
            MPI_Recv(&kicker, COUNT_1, MPI_INT, field, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (player.id == kicker) {
                // kick to new location
                ball.x = rand() % LENGTH;
                ball.y = rand() % WIDTH;
                player.kicked += 1;
                if (DEBUG) printf("Ball kicked by %d to %d, %d\n", player.id, ball.x, ball.y);
                MPI_Send(&ball, COUNT_1, mpi_ball, field, tag, MPI_COMM_WORLD);
                MPI_Send(&player, COUNT_1, mpi_player, field, tag, MPI_COMM_WORLD); 
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}