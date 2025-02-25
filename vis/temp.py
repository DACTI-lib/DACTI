import os
import time

def rename_figures_with_suffix(folder_path):
    try:
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Filter for figure files (e.g., images like .png, .jpg, .jpeg, etc.)
        figure_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        figures = [f for f in files if os.path.splitext(f)[1].lower() in figure_extensions]

        # Sort figures by creation time
        figures.sort(key=lambda f: os.path.getctime(os.path.join(folder_path, f)))

        # Rename each figure with a numbered suffix
        for i, filename in enumerate(figures, start=1):
            base, ext = os.path.splitext(filename)
            new_filename = f"cylinder_acti_level_{i:04d}.png"
            os.rename(
                os.path.join(folder_path, filename),
                os.path.join(folder_path, new_filename)
            )
            print(f"Renamed '{filename}' to '{new_filename}'")

        print("Renaming completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_path = "vis/out"
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        rename_figures_with_suffix(folder_path)
    else:
        print("Invalid folder path. Please try again.")
