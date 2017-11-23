/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2016 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "ctest.h"

/*
 * This test checks single-way isend and irecv with derived datatypes.
 */

#ifdef DEBUG
#define debug_printf(str,...) do {        \
    fprintf(stdout, str, ## __VA_ARGS__); \
    fflush(stdout);                       \
} while (0)
#else
#define debug_printf(str,...)
#endif

#define NUM_OPS 10
#define COUNT 10        /* count of double */

double *sbuf = NULL, *rbuf = NULL, *packed_rbuf = NULL, *packed_sbuf = NULL;
int rank, nprocs;
MPI_Win sbuf_win = MPI_WIN_NULL, rbuf_win = MPI_WIN_NULL;
MPI_Comm comm_world = MPI_COMM_NULL;
MPI_Datatype dtype = MPI_DOUBLE;
int dtype_size = sizeof(double);
MPI_Aint dtype_extent = sizeof(double);
int dtype_extent_ne = 1, dtype_size_ne = 1;
int ITER = 10;

typedef enum {
    CTEST_DTYPE_CONTIG,
    CTEST_DTYPE_VECTOR,
    CTEST_DTYPE_HVECTOR,
    CTEST_DTYPE_INDEXED,
    CTEST_DTYPE_HINDEXED,
    CTEST_DTYPE_IDX_BLK,
    CTEST_DTYPE_HIDX_BLK,
    CTEST_DTYPE_SUBARRAY,
    CTEST_DTYPE_STRUCT,
    CTEST_DTYPE_RESIZED,
    CTEST_DTYPE_DUP,
    CTEST_DTYPE_HVEC_VEC,
    CTEST_DTYPE_MAX
} CTEST_dtype_id_t;

static void create_datatype(CTEST_dtype_id_t dtype_id)
{
    MPI_Aint lb = 0;

    switch (dtype_id) {
    case CTEST_DTYPE_VECTOR:
        MPI_Type_vector(50, 2, 2, MPI_DOUBLE, &dtype);
        debug_printf("Generated vector\n");
        break;
    case CTEST_DTYPE_HVECTOR:
        MPI_Type_create_hvector(50, 2, 16, MPI_DOUBLE, &dtype);
        debug_printf("Generated hvector\n");
        break;
    case CTEST_DTYPE_INDEXED:
        {
            int array_blens[4] = { 5, 5, 2, 5 };
            int array_disps[4] = { 0, 10, 20, 30 };
            MPI_Type_indexed(4, array_blens, array_disps, MPI_DOUBLE, &dtype);
            debug_printf("Generated indexed\n");
        }
        break;
    case CTEST_DTYPE_HINDEXED:
        {
            int array_blens[4] = { 5, 5, 2, 5 };
            MPI_Aint array_disps[4] = { 0, 80, 160, 240 };
            MPI_Type_create_hindexed(4, array_blens, array_disps, MPI_DOUBLE, &dtype);
            debug_printf("Generated hindexed\n");
        }
        break;
    case CTEST_DTYPE_IDX_BLK:
        {
            int array_disps[4] = { 0, 10, 20, 30 };
            MPI_Type_create_indexed_block(4, 5, array_disps, MPI_DOUBLE, &dtype);
            MPI_Type_commit(&dtype);
            debug_printf("Generated indexed_block\n");
        }
        break;
    case CTEST_DTYPE_HIDX_BLK:
        {
            MPI_Aint array_disps[4] = { 0, 80, 160, 2400 };
            MPI_Type_create_hindexed_block(4, 5, array_disps, MPI_DOUBLE, &dtype);
            debug_printf("Generated hindexed_block\n");
        }
        break;
    case CTEST_DTYPE_SUBARRAY:
        {
            int array_sizes[3] = { 16, 16, 16 };
            int array_subsizes[3] = { 2, 2, 2 };
            int array_starts[3] = { 2, 2, 2 };
            MPI_Type_create_subarray(3, array_sizes, array_subsizes, array_starts,
                                     MPI_ORDER_C, MPI_DOUBLE, &dtype);
            debug_printf("Generated subarray\n");
        }
        break;
    case CTEST_DTYPE_STRUCT:
        {
            int array_blens[2] = { 2, 2 };
            MPI_Aint array_disps[2] = { 0, 256 };
            MPI_Datatype array_types[2];
            MPI_Datatype dt1;
            MPI_Type_vector(2, 2, 2, MPI_DOUBLE, &dt1);

            array_types[0] = dt1;
            array_types[1] = MPI_DOUBLE;
            MPI_Type_create_struct(2, array_blens, array_disps, array_types, &dtype);
            MPI_Type_free(&dt1);
            debug_printf("Generated struct\n");
        }
        break;
    case CTEST_DTYPE_RESIZED:
        {
            MPI_Datatype dt1;
            MPI_Type_contiguous(10, MPI_DOUBLE, &dt1);
            MPI_Type_create_resized(dt1, 0, 256, &dtype);
            MPI_Type_free(&dt1);
            debug_printf("Generated resized\n");
        }
        break;
    case CTEST_DTYPE_DUP:
        {
            MPI_Datatype dt1;
            MPI_Type_contiguous(100, MPI_DOUBLE, &dt1);
            MPI_Type_dup(dt1, &dtype);
            MPI_Type_free(&dt1);
            debug_printf("Generated dup\n");
        }
        break;
    case CTEST_DTYPE_HVEC_VEC:
        {
            MPI_Datatype dt1;
            MPI_Type_vector(2, 2, 2, MPI_DOUBLE, &dt1);
            MPI_Type_hvector(5, 1, 128, dt1, &dtype);
            MPI_Type_free(&dt1);
            debug_printf("Generated hvector-vector\n");
        }
        break;
    default:
        MPI_Type_contiguous(100, MPI_DOUBLE, &dtype);
        debug_printf("Generated contig\n");
        break;
    }

    MPI_Type_commit(&dtype);
    MPI_Type_size(dtype, &dtype_size);
    MPI_Type_get_extent(dtype, &lb, &dtype_extent);
    dtype_extent_ne = dtype_extent / sizeof(double);
    dtype_size_ne = dtype_size / sizeof(double);

    debug_printf("size %d, extent %d, nelem in size %d, nelem in extent %d\n",
                 dtype_size, dtype_extent, dtype_size_ne, dtype_extent_ne);
}

static void free_datatype()
{
    if (dtype != MPI_DOUBLE && dtype != MPI_DATATYPE_NULL)
        MPI_Type_free(&dtype);
}

static int check_stat(MPI_Status stat, int peer, int tag)
{
    int errs = 0;

    if (stat.MPI_TAG != tag) {
        fprintf(stderr, "[%d] stat.MPI_TAG %d != %d\n", rank, stat.MPI_TAG, tag);
        fflush(stderr);
        errs++;
    }
    if (stat.MPI_SOURCE != peer) {
        fprintf(stderr, "[%d] stat.MPI_SOURCE %d != %d\n", rank, stat.MPI_SOURCE, peer);
        fflush(stderr);
        errs++;
    }
    if (stat.MPI_ERROR != MPI_SUCCESS) {
        fprintf(stderr, "[%d] stat.MPI_ERROR 0x%x != MPI_SUCCESS 0x%x\n",
                rank, stat.MPI_ERROR, MPI_SUCCESS);
        fflush(stderr);
        errs++;
    }

    return errs;
}

static void pack_buf(double *ddtbuf, double *packbuf)
{
    int pos = 0;
    MPI_Pack(ddtbuf, COUNT, dtype, packbuf, dtype_size * COUNT, &pos, comm_world);
}

static int run_test(void)
{
    int i, x, c, e, errs = 0, errs_total = 0;
    int peer;
    MPI_Request req;
    MPI_Status stat;

    if (rank % 2)       /* receive only */
        peer = (rank - 1 + nprocs) % nprocs;
    else        /* send only */
        peer = (rank + 1) % nprocs;

    for (x = 0; x < ITER; x++) {

        if (rank % 2) { /* receive only */
            for (i = 0; i < NUM_OPS; i++) {
#ifdef USE_RECV_DDT
                MPI_Irecv(&rbuf[i * dtype_extent_ne * COUNT], COUNT, dtype, peer, i, comm_world,
                          &req);
#else
                MPI_Irecv(&rbuf[i * dtype_size_ne * COUNT], COUNT * dtype_size_ne,
                          MPI_DOUBLE, peer, i, comm_world, &req);
#endif
                stat.MPI_ERROR = MPI_SUCCESS;
                MPI_Wait(&req, &stat);

                /* Check completed receive.
                 * To avoid per datatype traversal, we always compare the packed buffer. */
#ifdef USE_RECV_DDT
                pack_buf(&rbuf[i * dtype_extent_ne * COUNT], packed_rbuf);
#else
                memcpy(packed_rbuf, &rbuf[i * dtype_size_ne * COUNT], COUNT * dtype_size);
#endif
#ifdef USE_SEND_DDT
                pack_buf(&sbuf[i * dtype_extent_ne * COUNT], packed_sbuf);
#else
                memcpy(packed_sbuf, &sbuf[i * dtype_size_ne * COUNT], COUNT * dtype_size);
#endif
                for (c = 0; c < COUNT; c++) {
                    for (e = 0; e < dtype_size_ne; e++) {
                        int off = c * dtype_size_ne + e;
                        if (CTEST_double_diff(packed_rbuf[off], packed_sbuf[off])) {
                            fprintf(stderr, "[%d] rbuf[%d] %.1lf != %.1lf\n",
                                    rank, off, packed_rbuf[off], packed_sbuf[off]);
                            fflush(stderr);
                            MPI_Abort(MPI_COMM_WORLD, -1);
                            errs++;
                        }
                    }
                }
                errs += check_stat(stat, peer, i);
            }
        }
        else {  /* send only */
            for (i = 0; i < NUM_OPS; i++) {
#ifdef USE_SEND_DDT
                MPI_Isend(&sbuf[i * dtype_extent_ne * COUNT], COUNT, dtype, peer, i, comm_world,
                          &req);
#else
                MPI_Isend(&sbuf[i * dtype_size_ne * COUNT], COUNT * dtype_size_ne,
                          MPI_DOUBLE, peer, i, comm_world, &req);
#endif
                MPI_Wait(&req, &stat);
            }
        }

        /* reset */
        for (i = 0; i < NUM_OPS * COUNT * dtype_extent_ne; i++)
            rbuf[i] = rbuf[i] * -1;
    }

    MPI_Allreduce(&errs, &errs_total, 1, MPI_INT, MPI_SUM, comm_world);
    return errs_total;
}

int main(int argc, char *argv[])
{
    int i, errs = 0;
    MPI_Info info = MPI_INFO_NULL;
    MPI_Comm shm_comm = MPI_COMM_NULL;
    CTEST_dtype_id_t dtype_id = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs < 2 || nprocs % 2) {
        fprintf(stderr, "Please run using power of two number of processes\n");
        goto exit;
    }

    MPI_Info_create(&info);

    /* Register as shared buffer in Casper. */
    MPI_Info_set(info, (char *) "shmbuf_regist", (char *) "true");
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, info, &shm_comm);

    MPI_Info_set(info, (char *) "wildcard_used", (char *) "none");
    MPI_Info_set(info, (char *) "datatype_used", (char *) "derived");
#ifdef USE_OFFLOAD_MIN_MSGSZ
    MPI_Info_set(info, (char *) "offload_min_msgsz", (char *) "1");
#endif
    MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &comm_world);
    MPI_Barrier(comm_world);

    for (dtype_id = 0; dtype_id < CTEST_DTYPE_MAX; dtype_id++) {
        create_datatype(dtype_id);

        MPI_Win_allocate_shared(dtype_extent * NUM_OPS * COUNT, MPI_DOUBLE,
                                MPI_INFO_NULL, shm_comm, &sbuf, &sbuf_win);
        MPI_Win_allocate_shared(dtype_extent * NUM_OPS * COUNT, MPI_DOUBLE,
                                MPI_INFO_NULL, shm_comm, &rbuf, &rbuf_win);

        packed_sbuf = malloc(dtype_size * COUNT);
        packed_rbuf = malloc(dtype_size * COUNT);

        for (i = 0; i < NUM_OPS * COUNT * dtype_extent_ne; i++) {
            sbuf[i] = 1.0 * i;
            rbuf[i] = sbuf[i] * -1;
        }
        memset(packed_sbuf, 0, dtype_size * COUNT);
        memset(packed_rbuf, 0, dtype_size * COUNT);

        errs += run_test();

        if (sbuf_win != MPI_WIN_NULL)
            MPI_Win_free(&sbuf_win);
        if (rbuf_win != MPI_WIN_NULL)
            MPI_Win_free(&rbuf_win);
        if (packed_sbuf)
            free(packed_sbuf);
        if (packed_rbuf)
            free(packed_rbuf);
        free_datatype();
    }

  exit:
    if (rank == 0)
        CTEST_report_result(errs);

    if (info != MPI_INFO_NULL)
        MPI_Info_free(&info);

    if (shm_comm != MPI_COMM_NULL)
        MPI_Comm_free(&shm_comm);
    if (comm_world != MPI_COMM_NULL)
        MPI_Comm_free(&comm_world);

    MPI_Finalize();
    return 0;
}
