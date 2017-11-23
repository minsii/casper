/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspg.h"

static inline int dtype_dup_regist_impl(CSP_cwp_dtype_dup_regist_pkt_t * dup_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Datatype newtype = MPI_DATATYPE_NULL;
    MPI_Datatype g_oldtype = MPI_DATATYPE_NULL;

    /* Receive extra parameter from user. */
    CSP_CALLMPI(JUMP, PMPI_Recv(&g_oldtype, sizeof(MPI_Datatype),
                                MPI_BYTE, dup_pkt->user_local_root, CSP_CWP_PARAM_TAG,
                                CSP_PROC.local_comm, MPI_STATUS_IGNORE));

    /* Create new datatype */
    CSP_CALLMPI(JUMP, PMPI_Type_dup(g_oldtype, &newtype));
    CSP_ASSERT(newtype != MPI_DATATYPE_NULL);
    CSPG_DBG_PRINT("DTYPE: dup 0x%lx (oldtype 0x%lx)\n", (MPI_Aint) newtype, (MPI_Aint) g_oldtype);

    /* Send back the new datatype handle. */
    CSP_CALLMPI(JUMP, PMPI_Send(&newtype, sizeof(MPI_Datatype), MPI_BYTE,
                                dup_pkt->user_local_root, CSP_CWP_PARAM_TAG, CSP_PROC.local_comm));
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* *INDENT-OFF* */
CSPG_DEFINE_GENERIC_CWP_ROOT_HANDLER(dtype_dup_regist)
CSPG_DEFINE_GENERIC_CWP_HANDLER(dtype_dup_regist)
/* *INDENT-ON* */
