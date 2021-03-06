/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2014 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "ctest.h"

#define NUM_OPS 5
#define CHECK
#define OUTPUT_FAIL_DETAIL

double *winbuf = NULL;
double locbuf[NUM_OPS], checkbuf[NUM_OPS];
int rank, nprocs;
MPI_Win win = MPI_WIN_NULL;
int ITER = 10;

static void reset_win()
{
    int i;

    MPI_Win_lock_all(0, win);
    for (i = 0; i < NUM_OPS; i++) {
        winbuf[i] = 0.0;
    }
    MPI_Win_unlock_all(win);
}

static void change_data(int nop, int x)
{
    int i;
    for (i = 0; i < nop; i++) {
        locbuf[i] = 1.0 * (x + 1) * (i + 1);
    }
}

static int check_data_all(int nop)
{
    int errs = 0;
    /* note that it is in an epoch */
    int dst, i;

    memset(checkbuf, 0, NUM_OPS * sizeof(double));

    for (dst = 0; dst < nprocs; dst++) {
        MPI_Get(checkbuf, nop, MPI_DOUBLE, dst, 0, nop, MPI_DOUBLE, win);
        MPI_Win_flush(dst, win);

        for (i = 0; i < nop; i++) {
            if (CTEST_precise_double_diff(checkbuf[i], locbuf[i])) {
                fprintf(stderr, "[%d] winbuf[%d] %.1lf != %.1lf\n", dst, i, checkbuf[i], locbuf[i]);
                errs++;
            }
        }
    }

#ifdef OUTPUT_FAIL_DETAIL
    if (errs > 0) {
        CTEST_print_double_array(locbuf, nop, "locbuf");
        CTEST_print_double_array(checkbuf, nop, "winbuf");
    }
#endif

    return errs;
}

static int check_data(int nop, int dst)
{
    int errs = 0;
    /* note that it is in an epoch */
    int i;

    memset(checkbuf, 0, NUM_OPS * sizeof(double));

    MPI_Get(checkbuf, nop, MPI_DOUBLE, dst, 0, nop, MPI_DOUBLE, win);
    MPI_Win_flush(dst, win);

    for (i = 0; i < nop; i++) {
        if (CTEST_precise_double_diff(checkbuf[i], locbuf[i])) {
            fprintf(stderr, "[%d] winbuf[%d] %.1lf != %.1lf\n", dst, i, checkbuf[i], locbuf[i]);
            errs++;
        }
    }

#ifdef OUTPUT_FAIL_DETAIL
    if (errs > 0) {
        CTEST_print_double_array(locbuf, nop, "locbuf");
        CTEST_print_double_array(checkbuf, nop, "winbuf");
    }
#endif

    return errs;
}

/* check lock_all/acc(all) & flush_local_all + (NOP * acc(all)) & flush_all/unlock_all */
static int run_test1(int nop)
{
    int i, x, errs = 0;
    int dst;

    if (rank == 0) {
        for (x = 0; x < ITER; x++) {
            change_data(nop, x);

            MPI_Win_lock_all(0, win);

            /* enable load balancing */
            for (dst = 0; dst < nprocs; dst++)
                MPI_Accumulate(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_MAX, win);

            /* flush local so that I can change local buffers */
            MPI_Win_flush_local_all(win);

            change_data(nop, x + ITER);

            /* max does not need ordering */
            for (dst = 0; dst < nprocs; dst++) {
                for (i = 0; i < nop; i++) {
                    MPI_Accumulate(&locbuf[i], 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, MPI_MAX, win);
                }
            }
            /* still need flush before checking result on the target side */
            MPI_Win_flush_all(win);

            errs += check_data_all(nop);

            MPI_Win_unlock_all(win);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&errs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return errs;
}

/* check lock/acc & flush_local + (NOP * acc) & flush/unlock */
static int run_test2(int nop)
{
    int i, x, errs = 0;
    int dst;

    if (rank == 0) {
        dst = (rank + 1) % nprocs;

        for (x = 0; x < ITER; x++) {
            change_data(nop, x);

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, dst, 0, win);

            /* enable load balancing */
            MPI_Accumulate(&locbuf[0], 1, MPI_DOUBLE, dst, 0, 1, MPI_DOUBLE, MPI_MAX, win);

            /* flush local so that I can change local buffers */
            MPI_Win_flush_local(dst, win);

            change_data(nop, x + ITER);

            for (i = 0; i < nop; i++) {
                MPI_Accumulate(&locbuf[i], 1, MPI_DOUBLE, dst, i, 1, MPI_DOUBLE, MPI_MAX, win);
            }

            /* still need flush before checking result on the target side */
            MPI_Win_flush(dst, win);

            errs += check_data(nop, dst);

            MPI_Win_unlock(dst, win);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&errs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return errs;
}

int main(int argc, char *argv[])
{
    int size = NUM_OPS;
    int errs = 0;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least 2 processes\n");
        goto exit;
    }

#ifdef TEST_EPOCHS_USED_LOCKALL
    MPI_Info_create(&info);
    MPI_Info_set(info, (char *) "epochs_used", (char *) "lockall");
#endif

    /* size in byte */
    MPI_Win_allocate(sizeof(double) * NUM_OPS, sizeof(double), info, MPI_COMM_WORLD, &winbuf, &win);

    reset_win();
    MPI_Barrier(MPI_COMM_WORLD);
    errs = run_test1(size);
    if (errs)
        goto exit;

#ifndef TEST_EPOCHS_USED_LOCKALL
    /* skip single lock test if checks with lockall hint */
    reset_win();
    MPI_Barrier(MPI_COMM_WORLD);
    errs = run_test2(size);
    if (errs)
        goto exit;
#endif

  exit:
    if (rank == 0)
        CTEST_report_result(errs);

    if (info != MPI_INFO_NULL)
        MPI_Info_free(&info);
    if (win != MPI_WIN_NULL)
        MPI_Win_free(&win);

    MPI_Finalize();

    return 0;
}
