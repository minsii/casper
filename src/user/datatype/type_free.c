/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_free_impl(MPI_Datatype * datatype)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype *g_dtypes = NULL;
    MPI_Request *reqs = NULL;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_free_pkt_t *ddt_free_pkt = &pkt.u.fnc_dtype_free;
    int i, lrank = 0;

    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));
    g_dtypes = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Datatype));

    /* Send command to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_FREE, &pkt);
    ddt_free_pkt->user_local_root = lrank;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: free datatype 0x%lx\n", (MPI_Aint) * datatype);

    /* Scatter the ghost handle to each ghost process */
    reqs = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Request));
    for (i = 0; i < CSP_ENV.num_g; i++) {
        mpi_errno = CSPU_datatype_get_g_handle(*datatype, i, &g_dtypes[i]);
        CSP_CHKMPIFAIL_JUMP(mpi_errno);

        CSP_CALLMPI(JUMP, PMPI_Isend(&g_dtypes[i], sizeof(MPI_Datatype),
                                     MPI_BYTE, CSP_PROC.user.g_lranks[i], CSP_CWP_PARAM_TAG,
                                     CSP_PROC.local_comm, &reqs[i]));
    }
    CSP_CALLMPI(JUMP, PMPI_Waitall(CSP_ENV.num_g, reqs, MPI_STATUS_IGNORE));

    /* Locally free datatype */
    for (i = 0; i < CSP_ENV.num_g; i++)
        CSPU_datatype_g_hash_remove(&CSPU_datatype_db.g_dtype_hashs[i], *datatype);
    CSP_CALLMPI(JUMP, PMPI_Type_free(datatype));

  fn_exit:
    if (g_dtypes)
        free(g_dtypes);
    if (reqs)
        free(reqs);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPI_Type_free(MPI_Datatype * datatype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_free(datatype);

    mpi_errno = type_free_impl(datatype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
