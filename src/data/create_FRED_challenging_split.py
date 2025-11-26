import os

# download split files from https://github.com/miccunifi/FRED/tree/main/dataset_splits/challenging

def create_symlinks(base_dir=os.getcwd()+'/FRED/', output_dir='FRED_challenging', train_list='train.txt', test_list='test.txt'):
    src_train = os.path.join(base_dir, 'train')
    src_test = os.path.join(base_dir, 'test')

    out_train = os.path.join(output_dir, 'train')
    out_test = os.path.join(output_dir, 'test')

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    def process_list(list_path, src_root, out_root):
        with open(list_path, 'r') as f:
            # remove whitespace, leading and trailing slashes
            folders = [line.strip().strip('/').strip() for line in f if line.strip()]
        for folder in folders:
            found_path = None
            candidate = os.path.join(src_root, folder)
            if os.path.exists(candidate):
                found_path = candidate
            else:
                alt_root = src_test if src_root == src_train else src_train
                alt_candidate = os.path.join(alt_root, folder)
                if os.path.exists(alt_candidate):
                    found_path = alt_candidate

            if found_path:
                link_path = os.path.join(out_root, folder)
                try:
                    if os.path.lexists(link_path):
                        os.remove(link_path)
                    os.symlink(found_path, link_path, target_is_directory=True)
                    print(f"[Link] {found_path} → {link_path}")
                except OSError as e:
                    print(f"[Error] Could not create symlink for {folder}: {e}")
            else:
                print(f"[Warning] Folder not found in FRED/train or FRED/test: {folder}")

    process_list(train_list, src_train, out_train)
    process_list(test_list, src_test, out_test)

if __name__ == '__main__':
    create_symlinks(train_list='dataset_splits/challenging/challenging_train_split.txt', test_list='dataset_splits/challenging/challenging_test_split.txt')