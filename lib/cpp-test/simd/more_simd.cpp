
#include <sys/types.h>

#define SIZE (33333333)

struct Foo { int v[SIZE]; }; // __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));

long sum(Foo * v)
{
  long s = 0;
  for (size_t i = 0; i < SIZE; i++) s += v->v[i];
  return s;
}


int main(int argc, char *arv[]) {

  // int v[SIZE] = {1, 3, 5, 3, 1};
  Foo f;
  sum(&f);

  return 0;
}
