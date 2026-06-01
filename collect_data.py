import os
import cv2
import shutil

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100

# Support digits 0-9 and uppercase letters A-Z as the standard set
SUPPORTED_CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def get_existing_classes():
    """Return a list of classes (character strings) that already have data."""
    existing = []
    if os.path.exists(DATA_DIR):
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path) and len(os.listdir(item_path)) > 0:
                existing.append(item.upper())
    return sorted(existing)


def collect_for_class(cap, class_name):
    """Collect dataset_size images for a single class."""
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\nCollecting data for Class (Character): {class_name}')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return
        cv2.putText(frame, f'Class: {class_name}. Ready? Press "Q" !', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return
        cv2.putText(frame, f'Class: {class_name} - {counter + 1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

    print(f'Done collecting {dataset_size} images for class: {class_name}')


def show_menu():
    """Display the main menu and return the user's choice."""
    existing = get_existing_classes()

    print("\n" + "=" * 50)
    print(" HAND SIGN DATA COLLECTION TOOL")
    print("=" * 50)

    if existing:
        print(f"\nExisting data found for {len(existing)} class(es):")
        for class_name in existing:
            class_dir = os.path.join(DATA_DIR, class_name)
            num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  Class '{class_name}' ({num_images} images)")
    else:
        print("\nNo existing data found.")

    missing = [cls for cls in SUPPORTED_CLASSES if cls not in existing]
    if missing:
        print(f"\nMissing default classes ({len(missing)}):")
        print(f"  {', '.join(missing)}")

    print("\n--- OPTIONS ---")
    print("1. Erase ALL data and recollect everything from scratch")
    print("2. Edit a specific class (re-collect its images)")
    print("3. Add a new custom class / character")
    print("4. Collect ONLY missing default classes (skip existing)")
    print("5. Exit")
    print()
    choice = input("Enter your choice (1-5): ").strip()
    return choice


def main():
    while True:
        choice = show_menu()

        if choice == '1':
            # Erase all and start over
            confirm = input("WARNING: This will DELETE all existing images. Type 'yes' to confirm: ").strip().lower()
            if confirm != 'yes':
                print("Cancelled.")
                continue
            # Delete all directories inside data
            if os.path.exists(DATA_DIR):
                for item in os.listdir(DATA_DIR):
                    item_path = os.path.join(DATA_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"  Deleted: {item_path}")
            # Collect for all standard classes
            cap = cv2.VideoCapture(0)
            for cls in SUPPORTED_CLASSES:
                collect_for_class(cap, cls)
            cap.release()
            cv2.destroyAllWindows()
            print("\nAll default classes collected successfully!")

        elif choice == '2':
            # Edit a specific class
            existing = get_existing_classes()
            if not existing:
                print("No existing classes to edit.")
                continue
            print("\nAvailable classes to edit:")
            print(f"  {', '.join(existing)}")
            selected = input("\nEnter the character/class to re-collect (e.g. A): ").strip().upper()
            if not selected:
                continue
            if selected not in existing:
                print(f"Class '{selected}' does not exist.")
                continue
            # Clear old data for this class
            class_dir = os.path.join(DATA_DIR, selected)
            if os.path.isdir(class_dir):
                shutil.rmtree(class_dir)
                print(f"  Cleared old data for class '{selected}'")
            cap = cv2.VideoCapture(0)
            collect_for_class(cap, selected)
            cap.release()
            cv2.destroyAllWindows()

        elif choice == '3':
            # Add a custom class / character
            new_char = input("Enter the character/label for this custom class (e.g. G): ").strip().upper()
            if not new_char or len(new_char) == 0:
                print("Invalid label.")
                continue
            class_dir = os.path.join(DATA_DIR, new_char)
            if os.path.isdir(class_dir) and len(os.listdir(class_dir)) > 0:
                print(f"Class '{new_char}' already exists. Use option 2 to edit it.")
                continue
            cap = cv2.VideoCapture(0)
            collect_for_class(cap, new_char)
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDone! Dynamic pipeline will automatically recognize and include class '{new_char}'.")

        elif choice == '4':
            # Collect only missing classes
            existing = get_existing_classes()
            missing = [cls for cls in SUPPORTED_CLASSES if cls not in existing]
            if not missing:
                print("\nNo missing classes! All default characters already have data.")
                continue
            print(f"\nWill collect data for {len(missing)} missing class(es):")
            print(f"  {', '.join(missing)}")
            cap = cv2.VideoCapture(0)
            for cls in missing:
                collect_for_class(cap, cls)
            cap.release()
            cv2.destroyAllWindows()
            print("\nAll missing classes collected successfully!")

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == '__main__':
    main()
