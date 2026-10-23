#!/bin/bash
# Run every MDWF test found in the current directory (<build>/testing) and print PASS/FAIL from its exit status.
# Logs go to mdwf_test_logs/<test>.log. Usage, on a node with one GPU:
#   cd <build>/testing && bash ../scripts/mdwf/run_mdwf_tests.sh [test ...]
# LAUNCH sets the launcher, e.g. LAUNCH="srun -n 1" (default: none).
mkdir -p mdwf_test_logs
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=($(ls mdwf*Test 2>/dev/null))
[ ${#tests[@]} -eq 0 ] && { echo "no mdwf*Test executables here; build with 'make mdwftests'"; exit 1; }
failed=0
for t in "${tests[@]}"; do
    start=$SECONDS
    if $LAUNCH ./"$t" < /dev/null > "mdwf_test_logs/$t.log" 2>&1; then status=PASS; else status=FAIL; failed=$((failed + 1)); fi
    printf "%-40s %s  %5d s\n" "$t" "$status" $((SECONDS - start))
done
echo "${#tests[@]} tests, $failed failed"
[ $failed -eq 0 ]
