/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_contiguous_impl(int count, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_contig_regist_pkt_t *regist_pkt = &pkt.u.fnc_dtype_contig_regist;
    int i = 0, lrank = 0;
    MPI_Datatype *g_oldtypes = NULL;
    MPI_Request *reqs = NULL;

    /* Locally create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_contiguous(count, oldtype, newtype));

    /* Send command and static parameters to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));

    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_CONTIG_REGIST, &pkt);
    regist_pkt->user_local_root = lrank;
    regist_pkt->param.count = count;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: create contiguous 0x%lx (count %d, oldtype 0x%lx)\n",
                  (MPI_Aint) newtype, regist_pkt->param.count, (MPI_Aint) oldtype);

    /* Prepare other parameters and send to each ghost. */
    g_oldtypes = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Datatype));
    reqs = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Request));
    for (i = 0; i < CSP_ENV.num_g; i++) {
        mpi_errno = CSPU_datatype_get_g_handle(oldtype, i, &g_oldtypes[i]);
        CSP_CHKMPIFAIL_JUMP(mpi_errno);

        CSP_CALLMPI(JUMP, PMPI_Isend(&g_oldtypes[i], sizeof(MPI_Datatype),
                                     MPI_BYTE, CSP_PROC.user.g_lranks[i], CSP_CWP_PARAM_TAG,
                                     CSP_PROC.local_comm, &reqs[i]));
    }
    CSP_CALLMPI(JUMP, PMPI_Waitall(CSP_ENV.num_g, reqs, MPI_STATUS_IGNORE));

    mpi_errno = CSPU_datatype_regist_complete(*newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    if (g_oldtypes)
        free(g_oldtypes);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_contiguous(count, oldtype, newtype);

    mpi_errno = type_contiguous_impl(count, oldtype, newtype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
