/* --------------------------------------------------------------
    Two tables-driven exponent function.

    Home page: www.imach.uran.ru/exp

    Copyright 2001-2002 by Dr. Raul N.Shakirov, IMach of RAS(UB),
    Phillip S. Pang, Ph.D. Biochemistry and Molecular Biophysics.
    Columbia University. NYC. All Rights Reserved.

    Permission has been granted to copy, distribute and modify
    software in any context without fee, including a commercial
    application, provided that the aforesaid copyright statement
    is present here as well as exhaustive description of changes.

    THE SOFTWARE IS DISTRIBUTED "AS IS". NO WARRANTY OF ANY KIND
    IS EXPRESSED OR IMPLIED. YOU USE AT YOUR OWN RISK. THE AUTHOR
    WILL NOT BE LIABLE FOR DATA LOSS, DAMAGES, LOSS OF PROFITS OR
    ANY OTHER KIND OF LOSS WHILE USING OR MISUSING THIS SOFTWARE.
-------------------------------------------------------------- */
#include <math.h>       /* Header file for standard math functions */
#include <stdio.h>      /* Header file for standard io functions */
#include <cstdint>
#include "tick/base/math/t2exp.h"      /* Header file for t2exp function */

#ifdef  __cplusplus
extern "C" {
#endif/*__cplusplus*/

/* --------------------------------------------------------------
    Key constants:

    EXPMINVAL   Minimal absolute value of argument to use tables.
    EXPMAXVAL   Minimal absolute value of argument to use tables.
    EXPBITS1    Number of fractional bits for the raw count table.
    EXPBITS2    Number of fractional bits for the refining table.

    Method:     exp (i1 + i2) = exp (i1) * exp (i2)

                i1  contains an integer part of argument and
                    first EXPBITS1 bits of fractional part
                i2  contains successive EXPBITS2 bits of
                    fractional part.

    Tables are used for arguments in range -EXPMAXVAL..-EXPMINVAL;
    outside this range standard exp() function is in use.

    TIPS:   Setting of EXPMINVAL higher then 0 enables using of
            precise exp() function for arguments, which are close
            to 0. On the other side, default value EXPMINVAL = 0
            provides for faster computations.

            Sum EXPBITS1 + EXPBITS2 + 1 determines number of
            significant bits in the result. For higher precision,
            try increase EXPBITS1.
            For better performance, try decrease EXPBITS2. Do set
            EXPBITS2 higher then 11 because refining table must
            be small enough to fit entirely into the L1 cache.

            To reduce size of raw count table, decrease either
            EXPBITS1 or EXPMAXVAL.

    NOTE:   On changing any of EXPMAXVAL, EXPBITS1 or EXPBITS2,
            rebuild t2exp.inl by calling t2expinl().
            Changing of EXPMINVAL does not require for rebuilding.

    Ref. also explanations for function t2exp below.
-------------------------------------------------------------- */
#define EXPMINVAL (0)
#define EXPMAXVAL (64)
#define EXPBITS1  (6)
#define EXPBITS2  (11)

/* --------------------------------------------------------------
    Derivative constants
    MULT1      pow (2, EXPBITS1)
    MULT2      pow (2, EXPBITS2)
    EXPTABLE1  num of float elements in the raw count table.
    EXPTABLE2  num of float elements in the refining table.
    nmult12    multiplicator for t2exp()
-------------------------------------------------------------- */
#define MULT1     (2 << (EXPBITS1-1))
#define MULT2     (2 << (EXPBITS2-1))
#define EXPTABLE1 (MULT1 * EXPMAXVAL + 1)
#define EXPTABLE2 (MULT2)
static const float nmult12 = (-static_cast<std::int64_t>(MULT1 * MULT2));

/* --------------------------------------------------------------
    Include file with content of the raw count and refining tables.
    The file is built by function t2expinl().
-------------------------------------------------------------- */

#include "tick/base/math/t2exp.inl"

/* --------------------------------------------------------------
    Name:       t2exp

    Purpose:    Fast two table-driven exponent algorithm,
                effective for arg <= 0.

    Usage:      t2exp (arg)

    Domain:     Same as for standard exp() function
                (approximately -709 <= arg <= 709).

    Result:     Approximate exp of arg; if arg is outside the
                exp() domain, results are same as for standard
                exp() function - that is either 0 or INF.

    Method:     exp (i1 + i2) = exp (i1) * exp (i2)

                i1  contains an integer part of argument and
                    first EXPBITS1 bits of fractional part
                i2  contains successive EXPBITS2 bits of
                    fractional part.

    For argument within -EXPMAXVAL..-EXPMINVAL, the function
    gets value of exp (i1) from the raw count table exptable1
    and value of exp (i2) from the refining table exptable2.
    Max relative error is 1 / pow (2, EXPBITS1 + EXPBITS2 + 1).

    For argument outside -EXPMAXVAL..-EXPMINVAL, standard exp()
    function is called.

    NOTE:   For arguments outside of
            -2**(31-EXPBITS1-EXPBITS2)..2**(31-EXPBITS1-EXPBITS2)
            function works two times slower then exp() due to
            internal float-to-integer conversion overflow.
-------------------------------------------------------------- */

double t2exp(double arg) {
  std::int64_t i1;                   /* Components of arg */
  int i2;                   /* Components of arg */

  /* Extract valid bits to integer variable */

#if defined(_MSC_VER) && defined(_M_IX86)

  /*
      t2exp for VC++ and x86 processors uses conversion of
      double to nearest int (round mode).
  */

  __asm fld   nmult12
  __asm fmul  arg
  __asm fistp i1

#else /*_MSC_VER && _M_IX86*/

  /*
      t2exp for generic processors uses conversion of double
      to lower nearest int (floor mode).
      On x86 it's less efficient then conversion to nearest int.
  */

  i1 = static_cast<std::int64_t>(arg * nmult12);

#endif/*_MSC_VER && _M_IX86*/

  /* Divide valid bits to high and low parts */

  i2 = static_cast<int>(i1) & (EXPTABLE2 - 1);      /* Get low  BITS2 bits */
  i1 >>= EXPBITS2;                    /* Get high BITS1 bits */

  /* Check if the arg fits to EXPTABLE1 */

  if (static_cast<std::int64_t>(i1 - MULT1 * EXPMINVAL) <=
      static_cast<std::int64_t>(MULT1 * (EXPMAXVAL - EXPMINVAL))) {
    /* Use tables to get exp (i1) and exp (i2) */

    return (exptable1[static_cast<int>(i1)] * exptable2[i2]);
  }

  /* Use standard exp function. */

  return (exp(arg));
}

/* --------------------------------------------------------------
    Name:       t2expini

    Purpose:    Build tables for t2exp().

    Usage:      t2expini()

    Note:       Used for development purposes only!
-------------------------------------------------------------- */

void t2expini(void) {
  int i;

  for (i = 0; i < EXPTABLE1; i++) {
    exptable1[i] = static_cast<float>(exp(-(static_cast<double>(i)) / MULT1));
  }

  for (i = 0; i < EXPTABLE2; i++) {
#if defined(_MSC_VER) && defined(_M_IX86)

    /*
        t2exp() for VC++ and x86 processors uses conversion of
        double to nearest int (round mode) so we haven't add 0.5
        to exp() argument here.
    */
    exptable2[i] = static_cast<float>(exp(-static_cast<double>(i) /
                                          static_cast<std::int64_t>(MULT1 * MULT2)));

#else /*_MSC_VER && _M_IX86*/

    /*
        t2exp() for generic processors uses conversion of double
        to lower nearest int (floor mode) so we have add 0.5
        to exp() argument for better approximation.
    */
    exptable2[i] = static_cast<float>(exp(-(static_cast<double>(i) + 0.5) /
        static_cast<std::int64_t>(MULT1 * MULT2)));

#endif/*_MSC_VER && _M_IX86*/
  }
}

/* --------------------------------------------------------------
    Name:       t2expinl

    Purpose:    Print tables for t2exp() in format of file t2exp.inl.

    Usage:      t2expinl()

    Note:       Used for development purposes only!
-------------------------------------------------------------- */

void t2expinl(void) {
  int i;

  printf("/*\n"
             "    File with t2exp() tables for case\n"
             "    EXPTABLE1 = %d, MULT1 = %d, EXPTABLE2 = %d, MULT2 = %d\n"
             "    The file was generated by function t2expinl().\n*/\n",
         EXPTABLE1, MULT1, EXPTABLE2, MULT2);

  printf("\n#if EXPTABLE1 != %d || MULT1 != %d || "
             "EXPTABLE2 != %d || MULT2 != %d\n",
         EXPTABLE1, MULT1, EXPTABLE2, MULT2);

  printf("\nstatic float exptable1 [EXPTABLE1];");

  printf("\nstatic float exptable2 [EXPTABLE2];");

  printf("\n\n#else /*EXPTABLE && MULT*/\n");

  printf("\nstatic float exptable1 [EXPTABLE1] =\n{");

  for (i = 0;;) {
    if (i % 4 == 0) printf("\n   ");
    printf(" %.7ef",
           static_cast<float>(exp(-(static_cast<double>(i)) / MULT1)));
    if (++i < EXPTABLE1) printf(",");
    else
      break;
  }

  printf("\n};\n");

  printf("\nstatic float exptable2 [EXPTABLE2] =\n{");

  printf("\n#if defined(_MSC_VER) && defined(_M_IX86)\n");

  for (i = 0;;) {
    if (i % 4 == 0) printf("\n   ");
    /*
        t2exp() for VC++ and x86 processors uses conversion of
        double to nearest int (round mode) so we haven't add 0.5
        to exp() argument here.
    */
    printf(" %.7ef",
           static_cast<float>(exp(-(static_cast<double>(i)) /
               (static_cast<std::int64_t>(MULT1 * MULT2)))));
    if (++i < EXPTABLE2) printf(",");
    else
      break;
  }

  printf("\n\n#else /*_MSC_VER && _M_IX86*/\n");

  for (i = 0;;) {
    if (i % 4 == 0) printf("\n   ");
    /*
        t2exp() for generic processors uses conversion of double
        to lower nearest int (floor mode) so we have add 0.5
        to exp() argument for better approximation.
    */
    printf(" %.7ef",
           static_cast<float>(exp(-(static_cast<double>(i) + 0.5) /
               (static_cast<std::int64_t>(MULT1 * MULT2)))));
    if (++i < EXPTABLE2) printf(",");
    else
      break;
  }

  printf("\n\n#endif/*_MSC_VER && _M_IX86*/\n};\n");

  printf("\n#endif/*EXPTABLE && MULT*/\n");
}

#ifdef  __cplusplus
}
#endif/*__cplusplus*/
