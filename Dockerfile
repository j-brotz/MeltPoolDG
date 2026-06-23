ARG DEALII_IMAGE_VERSION="master"
FROM dealii/dealii:${DEALII_IMAGE_VERSION}-noble AS builder

ARG BUILD_TYPE="Release"
ARG COMPILE_JOBS=2
USER root

WORKDIR /app

COPY . .

RUN rm -rf build adaflo adaflo-build && \
git clone https://github.com/MeltPoolDG/adaflo.git && \
mkdir adaflo-build && \
cd adaflo-build && \
cmake \
-D CMAKE_CXX_STANDARD=20 \
-D CMAKE_CXX_FLAGS="-Werror -Wno-error=cpp -Wno-error=deprecated-declarations" \
-D BUILD_SHARED_LIBS=ON \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
../adaflo && \
make -j${COMPILE_JOBS} adaflo && \
cd /app && \
mkdir build && \
cd build && \
cmake \
-D CMAKE_CXX_STANDARD=20 \
-D CMAKE_CXX_FLAGS="-Werror -Wno-error=cpp -Wno-error=deprecated-declarations" \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D ADAFLO_LIB=/app/adaflo-build/ \
-D ADAFLO_INCLUDE=/app/adaflo/include/ \
-D MPDG_ENABLE_BENCHMARKING=OFF \
--verbose
.. && \
make -j${COMPILE_JOBS} 

ENV HWLOC_HIDE_ERRORS=2
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /app/build

ENTRYPOINT ["ctest", "--output-on-failure"]
