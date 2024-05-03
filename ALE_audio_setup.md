## Setup ALE with audio on Oscar

*Everywhere you see <cslogin> insert your cslogin*

1. Clone https://github.com/shayegano/Arcade-Learning-Environment and checkout the `audio_support` branch
2. Then clone https://github.com/libsdl-org/SDL-1.2.git
3. In that SDL-1.2 directory, run `Run './configure --prefix=/users/<cslogin> --libdir=/users/<cslogin>/bin/lib64; make; make install'`
4. Back in the `Arcade-Learning-Environment` directory, open `CMakeLists.txt` in a text editor
    - Around line 26, you'll see `if("SDL_FOUND" "AND" "SDL_VERSION_STRING"...)`, change that if statement to `if(SDL_FOUND AND SDL_VERSION_STRING VERSION_LESS 2.0.0)`
    - Someplace before the "if(USE_SDL)" line add these three lines:
    ```
        set(SDL_ROOT_DIR "/users/<cslogin>")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/users/<cslogin>/bin")
        set(SDL_INCLUDE_DIR "$/users/<cslogin>/include/SDL")
    ```
    - Save and close the file
4.5. `cd /users/<cslogin> && mkdir bin` then navigate back to the `Arcade-Learning-Environment` directory
5. Open an interactive session if you haven't (`interact -n 1 -m 16g`) and run `mkdir build && cd build`
6. `cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..`
7. `make -j 4`
8. Assuming everything worked, navigate to the project directory (this directory)
9. Enter into the container with `apptainer run --nv tensorflow-24.03-tf2-py3.simg`
10. Navigate back to the `Arcade-Learning-Environment` directory and run `pip install --user .`