/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */


#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"


int MPI_Get_elements_x(const MPI_Status * status, MPI_Datatype datatype, MPI_Count * count)
{
    int mpi_errno = MPI_SUCCESS;

    /* Skip internal processing when disabled */
    if (CSP_IS_DISABLED || CSP_IS_MODE_DISABLED(PT2PT))
        return PMPI_Get_elements_x(status, datatype, count);

    /* FIXME: how to get the count information from ghost status ?
     * Check status_set_elements. */
    CSP_ASSERT(0);

    return mpi_errno;
}
