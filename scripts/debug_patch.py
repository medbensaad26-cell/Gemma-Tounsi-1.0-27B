import importlib
import sys
sys.path.insert(0, '/tmp')
from apply_soup_compat_patch import find_gpu_py, patch_already_applied, apply_patch, verify_patch

gpu_py = find_gpu_py()
text = gpu_py.read_text(encoding='utf-8')
print('Already applied:', patch_already_applied(text))

patched = apply_patch(text)
try:
    verify_patch(patched)
    print('Verification PASSED')
except SystemExit:
    print('Verification FAILED')
    if 'if marker in name_lower:' in patched:
        print('Old pattern still present in patched text')
    else:
        print('Old pattern NOT in patched text')
    if 're.escape(marker)' in patched:
        print('New pattern present in patched text')
    else:
        print('New pattern NOT in patched text')

# Show what the patched section looks like
for i, line in enumerate(patched.splitlines()):
    if 'size_markers' in line or '27b' in line or 're.escape' in line or 'name_lower' in line or 'for marker' in line or 'return size' in line:
        print(f'{i}: {line}')
