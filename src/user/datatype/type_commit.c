/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

static inline int type_commit_impl(MPI_Datatype * datatype)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype *g_dtypes = NULL, old_datatype = *datatype;
    MPI_Request *reqs = NULL;
    CSP_cwp_pkt_t pkt;
    CSP_cwp_dtype_commit_pkt_t *ddt_commit_pkt = &pkt.u.fnc_dtype_commit;
    int i, lrank = 0;
    char ack;

    /* Locally commit datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_commit(datatype));

    /* Standard does not define whether the handle value is changed in commit,
     * thus let us assume no change for now and fix it if any MPI impl breaks here.*/
    CSP_ASSERT(old_datatype == *datatype);

    CSP_CALLMPI(JUMP, PMPI_Comm_rank(CSP_PROC.local_comm, &lrank));
    g_dtypes = CSP_calloc(CSP_ENV.num_g, sizeof(MPI_Datatype));

    /* Send command to root ghost.
     * Do not mlock ghost because only one node. */
    CSP_cwp_init_pkt(CSP_CWP_FNC_DTYPE_COMMIT, &pkt);
    ddt_commit_pkt->user_local_root = lrank;

    mpi_errno = CSPU_cwp_issue(&pkt);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

    CSP_DBG_PRINT("DTYPE: commiot datatype 0x%lx\n", (MPI_Aint) * datatype);

    /* Scatter the ghost handle to each ghost process */
    reqs = CSP_calloc(CSP_ENV.num_g * 2, sizeof(MPI_Request));
    for (i = 0; i < CSP_ENV.num_g; i++) {
        mpi_errno = CSPU_datatype_get_g_handle(*datatype, i, &g_dtypes[i]);
        CSP_CHKMPIFAIL_JUMP(mpi_errno);

        CSP_CALLMPI(JUMP, PMPI_Isend(&g_dtypes[i], sizeof(MPI_Datatype),
                                     MPI_BYTE, CSP_PROC.user.g_lranks[i], CSP_CWP_PARAM_TAG,
                                     CSP_PROC.local_comm, &reqs[i * 2]));

        /* Need wait for the completion of commit on ghost processes. */
        CSP_CALLMPI(JUMP, PMPI_Irecv(&ack, 1, MPI_CHAR, CSP_PROC.user.g_lranks[i],
                                     CSP_CWP_PARAM_TAG, CSP_PROC.local_comm, &reqs[i * 2 + 1]));
    }
    CSP_CALLMPI(JUMP, PMPI_Waitall(CSP_ENV.num_g * 2, reqs, MPI_STATUS_IGNORE));

  fn_exit:
    if (g_dtypes)
        free(g_dtypes);
    if (reqs)
        free(reqs);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPI_Type_commit(MPI_Datatype * datatype)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Type_commit(datatype);

    /* NOTE: We cannot simply decode a derived datatype and send to the ghost side
     * to construct the same one. This is because it may contains a child derived
     * datatype, whose handler on the ghost process is impossible to be tracked by
     * either hashing the handler address or setting an attribute.
     *
     * See MPI standard 3.1 page 119 line 3:
     * If these were derived datatypes, then the returned datatypes are new datatype
     * objects, and the user is responsible for freeing these datatypes with MPI_TYPE_FREE.
     *
     * Page 119 line 6:
     * The committed state of returned derived datatypes is undefined, i.e., the
     * datatypes may or may not be committed. Furthermore, the content of attributes of
     * returned datatypes is undefined. */

    mpi_errno = type_commit_impl(datatype);
    CSP_CHKMPIFAIL_JUMP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
