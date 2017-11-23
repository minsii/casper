/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2017 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cspu.h"

int MPI_Type_struct(int count, int *array_of_blocklengths, MPI_Aint * array_of_displacements,
                    MPI_Datatype * array_of_types, MPI_Datatype * newtype)
{
    /* Deprecated function */
    return MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements,
                                  array_of_types, newtype);
}
