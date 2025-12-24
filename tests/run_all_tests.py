import sys
import time
import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

print("Running All Tests for JKJMiniNN")

tests = [
    ('test_basic.py', 'Basic functionality tests'),
    ('test_iris.py', 'Iris classification test'),
    ('test_digits.py', 'Digits classification test'),
    ('test_regression.py', 'Regression test'),
]

results = []

for test_file, description in tests:
    print(f"Running: {description}")
    print(f"File: {test_file}")

    start_time = time.time()
    test_path = os.path.join(os.path.dirname(__file__), test_file)

    try:
        with open(test_path, 'r') as f:
            code = f.read()
        exec(compile(code, test_path, 'exec'), globals())
        elapsed = time.time() - start_time
        results.append((test_file, 'PASSED', elapsed))
        print(f"{test_file} PASSED in {elapsed:.2f}s")
    except Exception:
        elapsed = time.time() - start_time
        results.append((test_file, 'FAILED', elapsed))
        print(f" {test_file} FAILED in {elapsed:.2f}s")
        import traceback
        print("Error:")
        print(traceback.format_exc())

print("Test Summary")

total_passed = sum(1 for _, status, _ in results if status == 'PASSED')
total_tests = len(results)

for test_file, status, elapsed in results:
    print(f" {test_file:30} {status:10} ({elapsed:.2f}s)")

print(f"Total: {total_passed}/{total_tests} tests passed")

if total_passed == total_tests:
    print(" All tests passed successfully!")
else:
    print(f" {total_tests - total_passed} test(s) failed")