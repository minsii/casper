/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2016 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cspg.h"

static inline int dtype_commit_impl(CSP_cwp_dtype_commit_pkt_t * dtype_commit_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype datatype_handle = MPI_DATATYPE_NULL;
    char ack;

    /* Receive the handle of my ug_comm from user root */
    CSP_CALLMPI(JUMP, PMPI_Recv(&datatype_handle, sizeof(MPI_Datatype), MPI_BYTE,
                                dtype_commit_pkt->user_local_root, CSP_CWP_PARAM_TAG,
                                CSP_PROC.local_comm, MPI_STATUS_IGNORE));

    CSPG_DBG_PRINT("DTYPE: commit datatype 0x%lx\n", (MPI_Aint) datatype_handle);
    CSP_CALLMPI(JUMP, PMPI_Type_commit(&datatype_handle));

    /* Send ack back thus the user knows it is ready to offload requests. */
    CSP_CALLMPI(JUMP, PMPI_Send(&ack, 1, MPI_CHAR, dtype_commit_pkt->user_local_root,
                                CSP_CWP_PARAM_TAG, CSP_PROC.local_comm));

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static inline int dtype_free_impl(CSP_cwp_dtype_free_pkt_t * dtype_free_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype datatype_handle = MPI_DATATYPE_NULL;

    /* Receive the handle of my ug_comm from user root */
    CSP_CALLMPI(JUMP, PMPI_Recv(&datatype_handle, sizeof(MPI_Datatype), MPI_BYTE,
                                dtype_free_pkt->user_local_root, CSP_CWP_PARAM_TAG,
                                CSP_PROC.local_comm, MPI_STATUS_IGNORE));

    CSPG_DBG_PRINT("DTYPE: free datatype 0x%lx\n", (MPI_Aint) datatype_handle);
    CSP_CALLMPI(JUMP, PMPI_Type_free(&datatype_handle));

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* *INDENT-OFF* */
CSPG_DEFINE_GENERIC_CWP_ROOT_HANDLER(dtype_free)
CSPG_DEFINE_GENERIC_CWP_HANDLER(dtype_free)

CSPG_DEFINE_GENERIC_CWP_ROOT_HANDLER(dtype_commit)
CSPG_DEFINE_GENERIC_CWP_HANDLER(dtype_commit)
/* *INDENT-ON* */

int CSPG_datatype_destory(void)
{
    return MPI_SUCCESS;
}

static void dtype_regist_handlers(void)
{
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_CONTIG_REGIST,
                                   CSPG_dtype_contig_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_VECTOR_REGIST,
                                   CSPG_dtype_vector_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_HVECTOR_REGIST,
                                   CSPG_dtype_hvector_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_IDX_BLK_REGIST,
                                   CSPG_dtype_idx_blk_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_HIDX_BLK_REGIST,
                                   CSPG_dtype_hidx_blk_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_INDEXED_REGIST,
                                   CSPG_dtype_indexed_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_HINDEXED_REGIST,
                                   CSPG_dtype_hindexed_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_STRUCT_REGIST,
                                   CSPG_dtype_struct_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_SUBARRAY_REGIST,
                                   CSPG_dtype_subarray_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_DARRAY_REGIST,
                                   CSPG_dtype_darray_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_RESIZED_REGIST,
                                   CSPG_dtype_resized_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_DUP_REGIST,
                                   CSPG_dtype_dup_regist_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_COMMIT, CSPG_dtype_commit_cwp_root_handler);
    CSPG_cwp_register_root_handler(CSP_CWP_FNC_DTYPE_FREE, CSPG_dtype_free_cwp_root_handler);

    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_CONTIG_REGIST,
                              CSPG_dtype_contig_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_VECTOR_REGIST,
                              CSPG_dtype_vector_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_HVECTOR_REGIST,
                              CSPG_dtype_hvector_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_IDX_BLK_REGIST,
                              CSPG_dtype_idx_blk_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_HIDX_BLK_REGIST,
                              CSPG_dtype_hidx_blk_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_INDEXED_REGIST,
                              CSPG_dtype_indexed_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_HINDEXED_REGIST,
                              CSPG_dtype_hindexed_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_STRUCT_REGIST,
                              CSPG_dtype_struct_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_SUBARRAY_REGIST,
                              CSPG_dtype_subarray_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_DARRAY_REGIST,
                              CSPG_dtype_darray_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_RESIZED_REGIST,
                              CSPG_dtype_resized_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_DUP_REGIST, CSPG_dtype_dup_regist_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_COMMIT, CSPG_dtype_commit_cwp_handler);
    CSPG_cwp_register_handler(CSP_CWP_FNC_DTYPE_FREE, CSPG_dtype_free_cwp_handler);
}

int CSPG_datatype_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Request *reqs = NULL;
    int i, local_rank = 0;
    MPI_Datatype local_predefined_table[CSP_DATATYPE_MAX], temp_buf[CSP_DATATYPE_MAX];

    CSP_datatype_fill_predefined_table(local_predefined_table);
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &local_rank));

    reqs = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Request));

    /* Bcast my predefined datatype table, and join the other ghosts' bcast. */
    for (i = 0; i < CSP_ENV.num_g; i++) {
        char *buf = (char *) temp_buf;
        if (i == local_rank)
            buf = (char *) local_predefined_table;
        CSP_CALLMPI(JUMP, PMPI_Ibcast(buf, CSP_DATATYPE_MAX * sizeof(MPI_Datatype),
                                      MPI_BYTE, i, CSP_PROC.local_comm, &reqs[i]));
    }
    CSP_CALLMPI(JUMP, PMPI_Waitall(CSP_ENV.num_g, reqs, MPI_STATUS_IGNORE));

    dtype_regist_handlers();

    CSP_DBG_PRINT("DTYPE: init done\n");

  fn_exit:
    free(reqs);
    return mpi_errno;
  fn_fail:
    /* Free global objects in main function. */
    goto fn_exit;
}
