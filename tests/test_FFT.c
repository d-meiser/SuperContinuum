#include <cgreen/cgreen.h>
#include <fft.h>

Ensure(need_meaningful_dmda_to_construct_fft) {
  NppFft fft =   nppCreateFftFromDMDA(0);
  assert_that(fft, is_null);
}

int main(int argc, char **argv) {
  TestSuite *suite = create_test_suite();
  add_test(suite, need_meaningful_dmda_to_construct_fft);
  return run_test_suite(suite, create_text_reporter());
}
