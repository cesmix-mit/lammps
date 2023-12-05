    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <time.h>
    // #include <chrono>
    // #include <iostream>

    #define DGEMM dgemm_
    #define DSYEV dsyev_
    
    extern "C" {
    void DGEMM(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *,
               double *, double *, int *);
    void DSYEV(char *, char *, int *, double *, int *, double *, double *, int *, int *);
    }

    class EnvClustering {
    public:
        double* desc;
        double* P;
        double* Lambda;
        double* Phi;
        double* PD;
        double* centroids;
        int* clusters;
        double* invSqDist;
        double* clusterProba;
        double* dPdD;
        double* dDdPD;
        double* dPddesc;

    public:
        EnvClustering(int nAtoms, int Mdesc, int nComponents, int nClusters) {
            desc = (double*)malloc(nAtoms * Mdesc * sizeof(double));
            P = (double*)malloc(Mdesc * nComponents * sizeof(double));
            Lambda = (double*)malloc(Mdesc * sizeof(double));
            Phi = (double*)malloc(Mdesc * Mdesc * sizeof(double));
            PD = (double*)malloc(nAtoms * nComponents * sizeof(double));
            centroids = (double*)malloc(nClusters * nComponents * sizeof(double));
            clusters = (int*)malloc(nAtoms * sizeof(int));
            invSqDist = (double*)malloc(nAtoms * nClusters * sizeof(double));
            clusterProba = (double*)malloc(nAtoms * nClusters * sizeof(double));
            dPdD = (double*)malloc(nAtoms * nClusters * sizeof(double));
            dDdPD = (double*)malloc(nAtoms * nClusters * nComponents * sizeof(double));
            dPddesc = (double*)malloc(nAtoms * nClusters * Mdesc * sizeof(double));
        }

        ~EnvClustering() {
            free(desc);
            free(P);
            free(Lambda);
            free(Phi);
            free(PD);
            free(centroids);
            free(clusters);
            free(invSqDist);
            free(clusterProba);
            free(dPdD);
            free(dDdPD);
            free(dPddesc);
        }

        void eigenDecomposition(double* desc, double* Lambda, double* Phi, int nAtoms, int Mdesc)
        {
            // double *Q = (double *) malloc(nAtoms*Mdesc*sizeof(double));
            double *A = (double *) malloc(Mdesc*Mdesc*sizeof(double));
            double *b = (double *) malloc(Mdesc*sizeof(double));

            // Calculate covariance matrix S = desc'*desc
            // double* S = (double*)malloc(Mdesc * Mdesc * sizeof(double));
            // for (int i = 0; i < Mdesc; i++) {
            //     for (int j = 0; j < Mdesc; j++) {
            //         S[i * Mdesc + j] = 0.0;
            //         for (int k = 0; k < nAtoms; k++) {
            //             S[i * Mdesc + j] += desc[k * Mdesc + i] * desc[k * Mdesc + j];
            //         }
            //     }
            // }

            // Calculate covariance matrix A = desc'*desc
            char chn = 'N';
            char cht = 'T';
            char chv = 'V';
            char chu = 'U';
            double alpha = 1.0, beta = 0.0;

            DGEMM(&cht, &chn, &Mdesc, &Mdesc, &nAtoms, &alpha, desc, &nAtoms, desc, &nAtoms, &beta, A, &Mdesc);

            int lwork = Mdesc * Mdesc;  // the length of the array work, lwork >= max(1,3*N-1)

            // for (int i=0; i<lwork; i++)
            //     A[i] = A[i]*(1.0/nAtoms);

            int info = 1;     // = 0:  successful exit
            double work[lwork];

            DSYEV(&chv, &chu, &Mdesc, A, &Mdesc, b, work, &lwork, &info);

            // order eigenvalues and eigenvectors from largest to smallest

            for (int j=0; j<Mdesc; j++)
                for (int i=0; i<Mdesc; i++)
                    Phi[i + Mdesc*(Mdesc-j-1)] = A[i + Mdesc*j];

            for (int i=0; i<Mdesc; i++)
                Lambda[(Mdesc-i-1)] = b[i];

            // DGEMM(&chn, &chn, &N, &ns, &ns, &alpha, S, &N, Phi, &ns, &beta, Q, &N);
            // for (int i=0; i<(N-1); i++)
            //     xij[i] = xij[i+1] - xij[i];
            // double area;
            // for (int m=0; m<ns; m++) {
            //    area = 0.0;
            //    for (int i=0; i<(N-1); i++)
            //        area += 0.5*xij[i]*(Q[i + N*m]*Q[i + N*m] + Q[i+1 + N*m]*Q[i+1 + N*m]);
            //        for (int i=0; i<ns; i++)
            //            Phi[i + ns*m] = Phi[i + ns*m]/sqrt(area);
            // }

            // enforce consistent signs for the eigenvectors
            // for (int m=0; m<Mdesc; m++) {
            //    if (Phi[m + Mdesc*m] < 0.0) {
            //        for (int i=0; i<Mdesc; i++)
            //            Phi[i + Mdesc*m] = -Phi[i + Mdesc*m];
            //        }
            // }

            free(A); free(b); // free(Q);
        }

        void PCA(double* desc, double* P, double* Lambda, double* Phi, int nAtoms, int Mdesc, int nComponents) {
            eigenDecomposition(desc, Lambda, Phi, nAtoms, Mdesc);

            // Calculate full projection matrix
            double* Pfull = (double*)malloc(Mdesc * Mdesc * sizeof(double));
            for (int i = 0; i < Mdesc; i++) {
                for (int j = 0; j < Mdesc; j++) {
                    Pfull[i * Mdesc + j] = Phi[i * Mdesc + j] * sqrt(fabs(Lambda[j]));
                }
            }
            
            // Keep the first nComponents number of dimensions
            for (int i = 0; i < Mdesc; i++) {
                for (int j = 0; j < nComponents; j++) {
                    P[i * nComponents + j] = Pfull[i * Mdesc + j];
                }
            }
            free(Pfull);
        }

        void saveProjMatToFile(double* P, int nComponents, int Mdesc, const char* filename) {
            FILE* file = fopen(filename, "w");
            if (file == NULL) {
                printf("Error opening file for writing.\n");
                return;
            }
            fprintf(file, "Mdesc: %d\n", Mdesc);
            fprintf(file, "nComponents: %d\n", nComponents);
            for (int i = 0; i < Mdesc; i++) {
                for (int j = 0; j < nComponents; j++) {
                    fprintf(file, "%f ", P[i * nComponents + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }

        void saveCentroidsToFile(const double* centroids, int nClusters, int nComponents, const char* filename) {
            FILE* file = fopen(filename, "w");
            if (file == NULL) {
                printf("Error opening centroids file for writing.\n");
                return;
            }
            fprintf(file, "nClusters: %d\n", nClusters);
            fprintf(file, "dimensions: %d\n", nComponents);

            for (int i = 0; i < nClusters; ++i) {
                for (int j = 0; j < nComponents; ++j) {
                    fprintf(file, "%f ", centroids[i * nComponents + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }

        void getInvDist(double* b, int nAtoms, int numDimensions, double* centroids, int nClusters, double* inverseDistances) {
            for (int i = 0; i < nAtoms; i++) {
                for (int j = 0; j < nClusters; j++) {
                    double dist = L2norm(&b[i * numDimensions], &centroids[i * numDimensions], numDimensions);
                    inverseDistances[i * nClusters + j] = 1.0 / dist;
                }
            }
        }

        void getSqInvDist(double* b, int nAtoms, int numDimensions, double* centroids, int nClusters, double* inverseDistances) {
            for (int i = 0; i < nAtoms; i++) {
                for (int j = 0; j < nClusters; j++) {
                    double distance = 0.0;
                    for (int k = 0; k < numDimensions; k++) {
                        double diff = b[i * numDimensions + k] - centroids[j * numDimensions + k];
                        distance += diff * diff;
                    }
                    inverseDistances[i * nClusters + j] = 1.0 / distance;
                }
            }
        }

        void getProba(const double* inverseSquareDistances, int nAtoms, int nClusters, double* probabilities) {
            for (int i = 0; i < nAtoms; i++) {
                double sumInverseSquareDistances = 0.0;
                for (int j = 0; j < nClusters; j++) {
                    sumInverseSquareDistances += inverseSquareDistances[i * nClusters + j];
                }
                for (int j = 0; j < nClusters; j++) {
                    probabilities[i * nClusters + j] = inverseSquareDistances[i * nClusters + j] / sumInverseSquareDistances;
                }
            }
        }

        // dPdD = (S-D)/S^2 = (1-Prob)/S
        // dPdD shape: (nAtoms, nClusters)
        void getdPdD(const double* inverseSquareDistances, int nAtoms, int nClusters, double* dPdD) {
            for (int i = 0; i < nAtoms; i++) {
                double sumInverseSquareDistances = 0.0;
                for (int j = 0; j < nClusters; j++) {
                    sumInverseSquareDistances += inverseSquareDistances[i * nClusters + j];
                }
                for (int j = 0; j < nClusters; j++) {
                    double D_ij = inverseSquareDistances[i * nClusters + j];
                    dPdD[i * nClusters + j] = (sumInverseSquareDistances - D_ij) / (sumInverseSquareDistances * sumInverseSquareDistances);
                }
            }
        }

        // dDdb = -2D^2(b-c), where b = desc * P = PD, c=centroids
        // dDdb shape: (nAtoms, nClusters, nComponents)
        void getdDdPD(const double* PD, const double* centroids, const double* inverseSquareDistances, int nAtoms, int nClusters, int nComponents, double* dDdPD) {
            for (int i = 0; i < nAtoms; i++) {
                for (int k = 0; k < nComponents; k++) {
                    double PD_ik = PD[i * nComponents + k];
                    for (int j = 0; j < nClusters; j++) {
                        double C_jk = centroids[j * nComponents + k];
                        double D_ij = inverseSquareDistances[i * nClusters + j];
                        dDdPD[(i * nClusters + j) * nComponents + k] = -2 * D_ij * D_ij * (PD_ik - C_jk);
                    }
                }
            }
        }

        // dPddesc = dPdD * dDdb * P
        // dPddesc shape: (nAtoms, nClusters, Mdesc)
        void getdPddesc(const double* dPdD, const double* dDdPD, const double* P, int nAtoms, int nClusters, int nComponents, int Mdesc, double* dPddesc) {
            for (int i = 0; i < nAtoms; i++) {
                for (int j = 0; j < nClusters; j++) {
                    for (int m = 0; m < Mdesc; m++) {
                        double dPdD_ij = dPdD[i * nClusters + j];
                        for (int k = 0; k < nComponents; k++) {
                            double dDdDP_ik = dDdPD[(i * nClusters + j) * nComponents + k];
                            double dPDik_ddesc = P[m * nComponents + k];
                            dPddesc[(i * nClusters + j) * Mdesc + m] += dPdD_ij * dDdDP_ik * dPDik_ddesc;
                        }
                    }
                }
            }
        }

        double L2norm(const double* a, const double* b, int dimensions) {
            double sum = 0.0;
            for (int i = 0; i < dimensions; ++i) {
                sum += pow(a[i] - b[i], 2);
            }
            return sqrt(sum);
        }
        // Function to assign each desc point to the nearest centroid
        void assignToClusters(const double* desc, const double* centroids, int* clusters, int nAtoms, int nClusters, int dimensions) {
            for (int i = 0; i < nAtoms; ++i) {
                double minDist = INFINITY;
                int clusterIndex = -1;
                for (int j = 0; j < nClusters; ++j) {
                    double dist = L2norm(&desc[i * dimensions], &centroids[j * dimensions], dimensions);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIndex = j;
                    }
                }
                clusters[i] = clusterIndex;
            }
        }
        // Function to update centroids based on assigned desc points
        void updateCentroids(const double* desc, const int* clusters, double* centroids, int nAtoms, int nClusters, int dimensions) {
            int* clusterSizes = (int*)calloc(nClusters, sizeof(int));
            double** sumCoordinates = (double**)malloc(nClusters * sizeof(double*));
            for (int i = 0; i < nClusters; ++i) {
                sumCoordinates[i] = (double*)calloc(dimensions, sizeof(double));
            }
            for (int i = 0; i < nAtoms; ++i) {
                int clusterIndex = clusters[i];
                clusterSizes[clusterIndex]++;
                for (int j = 0; j < dimensions; ++j) {
                    sumCoordinates[clusterIndex][j] += desc[i * dimensions + j];
                }
            }
            for (int i = 0; i < nClusters; ++i) {
                for (int j = 0; j < dimensions; ++j) {
                    if (clusterSizes[i] > 0) {
                        centroids[i * dimensions + j] = sumCoordinates[i][j] / clusterSizes[i];
                    }
                }
            }
            free(clusterSizes);
            for (int i = 0; i < nClusters; ++i) {
                free(sumCoordinates[i]);
            }
            free(sumCoordinates);
        }
        // Function to initialize centroids randomly
        void initializeCentroids(double* desc, double* centroids, int nAtoms, int nClusters, int dimensions) {
            for (int i = 0; i < nClusters; ++i) {
                int randomIndex = rand() % nAtoms;
                for (int j = 0; j < dimensions; ++j) {
                    centroids[i * dimensions + j] = desc[randomIndex * dimensions + j];
                }
            }
        }
        // K-Means clustering algorithm
        void kMeans(double* desc, double* centroids, int* clusters, int nAtoms, int nClusters, int dimensions) {

            initializeCentroids(desc, centroids, nAtoms, nClusters, dimensions);
            // Iterate until convergence
            for (int iteration = 0; iteration < 100; ++iteration) {
                // Assign desc points to the nearest centroid
                assignToClusters(desc, centroids, clusters, nAtoms, nClusters, dimensions);
                // Update centroids based on assigned desc points
                updateCentroids(desc, clusters, centroids, nAtoms, nClusters, dimensions);
                // Print cluster assignments (optional)
                // printf("Iteration %d - Cluster Assignments: ", iteration + 1);
                // for (int i = 0; i < nAtoms; ++i) {
                //     printf("(%f, %f): %d ", desc[i * dimensions], desc[i * dimensions + 1], clusters[i]);
                // }
                // printf("\n");
            }
        }
    };

    int main() {
        // srand((unsigned)time(NULL));
        srand(123);
        int nAtoms = 100;
        int Mdesc = 20;
        int nComponents = 2;
        int nClusters = 4;

        EnvClustering envClustering(nAtoms, Mdesc, nComponents, nClusters);

        for (int i = 0; i < nAtoms; i++) {
            for (int j = 0; j < Mdesc; j++) {
                envClustering.desc[i * Mdesc + j] = (double)rand() / RAND_MAX;
            }
        }

        envClustering.PCA(envClustering.desc, envClustering.P, envClustering.Lambda, envClustering.Phi, nAtoms, Mdesc, nComponents);

        const char* projMatFileName = "ProjMat.txt";
        envClustering.saveProjMatToFile(envClustering.P, nComponents, Mdesc, projMatFileName);
        
        char transA = 'N';
        char transB = 'N';
        int m = nAtoms;
        int n = nComponents;
        int k = Mdesc;
        double alpha = 1.0;
        double beta = 0.0;
        int lda = nAtoms;
        int ldb = Mdesc;
        int ldc = nAtoms;

        DGEMM(&transA, &transB, &m, &n, &k, &alpha, envClustering.desc, &lda, envClustering.P, &ldb, &beta, envClustering.PD, &ldc);
        
        envClustering.kMeans(envClustering.PD, envClustering.centroids, envClustering.clusters, nAtoms, nClusters, nComponents);

        const char* centroidFileName = "centroids.txt";
        envClustering.saveCentroidsToFile(envClustering.centroids, nClusters, nComponents, centroidFileName);

        printf("Final Centroids:\n");
        for (int i = 0; i < nClusters; ++i) {
            printf("Cluster %d: (%f, %f)\n", i, envClustering.centroids[i * nComponents], envClustering.centroids[i * nComponents + 1]);
        }
        // auto start = std::chrono::high_resolution_clock::now();
        envClustering.getSqInvDist(envClustering.PD, nAtoms, nComponents, envClustering.centroids, nClusters, envClustering.invSqDist);
        envClustering.getProba(envClustering.invSqDist, nAtoms, nClusters, envClustering.clusterProba);

        // printf("Cluster Probabilities:\n");
        // for (int i = 0; i < nAtoms; ++i) {
        //    printf("Data Point %d: ", i);
        //    for (int j = 0; j < nClusters; ++j) {
        //        printf("%f ", envClustering.clusterProba[i * nClusters + j]);
        //    }
        //    printf("\n");
        //}

        envClustering.getdPdD(envClustering.invSqDist, nAtoms, nClusters, envClustering.dPdD);
        envClustering.getdDdPD(envClustering.PD, envClustering.centroids, envClustering.invSqDist, nAtoms, nClusters, nComponents, envClustering.dDdPD);
        envClustering.getdPddesc(envClustering.dPdD, envClustering.dDdPD, envClustering.P, nAtoms, nClusters, nComponents, Mdesc, envClustering.dPddesc);
        
        // Generate random ddesc/dR, with shape (nAtoms, MDesc, 3*nNeighbors)
        int nNeighbors = 20;
        double* ddesc = (double*)malloc(nAtoms * Mdesc * 3*nNeighbors * sizeof(double));
        for (int i = 0; i < nAtoms; i++) {
            for (int j = 0; j < Mdesc; j++) {
                for (int k = 0; k < 3 * nNeighbors; k++) {
                    ddesc[(i * Mdesc + j) * 3 * nNeighbors + k] = (double)rand() / RAND_MAX;
                }
            }
        }

        // Calculate dP/dR = dP/ddesc * ddesc/dR
        // dPdR shape: (nAtoms, nClusters, 3*nNeighbors)
        // ToDo: Are the dimensions of dPdR multiplication correct?
        double* dPdR = (double*)malloc(nAtoms * nClusters * 3 * nNeighbors * sizeof(double));
        char transA2 = 'N';
        char transB2 = 'N';
        int m2 = nAtoms;
        int n2 = nClusters * 3 * nNeighbors;
        int k2 = Mdesc;
        double alpha2 = 1.0;
        double beta2 = 0.0;
        int lda2 = nAtoms;
        int ldb2 = Mdesc;
        int ldc2 = nAtoms;
        DGEMM(&transA2, &transB2, &m2, &n2, &k2, &alpha2, envClustering.dPddesc, &lda2, ddesc, &ldb2, &beta2, dPdR, &ldc2);
        
        // auto end = std::chrono::high_resolution_clock::now();
        // Calculate and print the elapsed time
        // std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        free(ddesc);
        free(dPdR);
        return 0;
    }