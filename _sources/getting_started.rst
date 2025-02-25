.. _Chap:getting_started:

Getting Started
===============

Download the code
-----------------
To download DACTI to your machine, make sure you have `Git <https://git-scm.com/>`_ installed. We recommend cloning the repository via SSH. If you haven't generated ssh key pairs yet, do 

#.  Generate a new SSH key pair and save them to the directory ``$HOME/.ssh``

    .. code-block:: shell

        ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519_github

    You can skip giving a passphrase by pressing :kbd:`Enter`. 

#.  Add the ``.pub`` key to your Github account
   
    .. note::
        The private key ``id_ed25519_github`` should be kept secret.

#.  Clone the repository

    .. code-block:: shell

        git clone git@github.com:DACTI-lib/DACTI.git


Building the code
-----------------
We use `CMake <https://cmake.org/>`_ to build the executable. The configuration can be found in ``DACTI/CMakeLists.txt``. First make sure you have installed CMake and a C++ compiler, such as `GCC <https://gcc.gnu.org>`_ or `Clang <https://clang.llvm.org>`_. You would also need to install `VTK <https://vtk.org>`_ for data output.

To build the code

#.  Create a build directory

    .. code-block:: shell

        cd /path/to/DACTI
        mkdir build; cd build

#.  Configure CMake

    .. code-block:: shell

        cmake ..

#.  Build executable

    .. code-block:: shell

       cmake --build . -j4 --target $EXAMPLE #-j4 for parallel build
    
    All available examples can be found in ``DACTI/src/examples``.

#.  Run

    .. code-block:: shell

        cd path/to/DACTI
        .build/$EXAMPLE cases/$CONFIG.toml

    ``.toml`` are configuration files for the examples, where you can define your own test case.


DACTI on ETH Euler Cluster
--------------------------
This section is for users who have access to the `Euler <https://scicomp.ethz.ch/wiki/Euler>`_ cluster at ETH Zurich. If you have never used Euler before, check out `Getting started with cluster <https://scicomp.ethz.ch/wiki/Getting_started_with_clusters>`_. 

To build and run DACTI on Euler

#.  Load the necessary modules

    .. code-block:: shell

        module load stack/2024-05 gcc/13.2.0 mesa/23.0.3 cmake

#.  Download and install VTK 

    .. code-block:: shell

        # create a directory to install VTK
        mkdir $HOME/.vtk

        # download from the internet
        wget https://github.com/Kitware/VTK/releases/tag/v9.3.0/VTK-9.3.0.zip
        unzip VTK-9.3.0.zip; cd VTK-9.3.0/

        # build and install
        mkdir build; cd build
        cmake -DCMAKE_INSTALL_PREFIX=~/.vtk -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_OPENGL_HAS_OSMESA=ON -DVTK_USE_X=OFF -DVTK_USE_SDL2=OFF ..
        sbatch -n1 --cpus-per-task=16 --wrap="cmake --build . -j16" # compile on compute node
        cmake --install .

    Here, VTK version 9.3.0 is recommended.

#.  Clone the repository

    .. code-block:: shell

        cd $HOME
        git clone git@github.com:DACTI-lib/DACTI.git

#.  Build the code

    .. code-block:: shell
        
        cd DACTI
        mkdir build; cd build
        cmake -DVTK_DIR=$HOME/.vtk/lib/cmake/vtk-9.3/ -DGLFW_USE_OSMESA=ON ..
        cmake --build . -j4 --target $EXAMPLE

    You might encounter the compilation error

    .. error:: 
        error: template-id not allowed for destructor

    To solve this, go to ``DACTI/build/_deps/libigl-src/include/igl/WindingNumberTree.h`` and replace line 217

    .. code-block:: cpp

        inline igl::WindingNumberTree<Point,DerivedV,DerivedF>::~WindingNumberTree<Point,DerivedV,DerivedF>()

    with 

    .. code-block:: cpp

        igl::WindingNumberTree<Point,DerivedV,DerivedF>::~WindingNumberTree()

#.  Run

    Submit a job using the Slurm system. More information can be found `here <https://scicomp.ethz.ch/wiki/Using_the_batch_system>`_. 
    
        
    
