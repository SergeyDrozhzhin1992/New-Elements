#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cmath> 
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <glpk.h>
#include <algorithm>
#include <cstdlib>
using namespace std;
const double EPS = 5.0E-2;

/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*Two Poisson processes: the first determines whether to make changes or not (the lambda1 parameter);
 *                       the second, in the case of changes being made first, determines to add a new view or replace an existing one (the lambda2 parameter).
 *                       The number of changes is limited to count_new_type
 *                    
 *     INPUT: 
 *         count_iter - the number of iterations;
 *         lambda1 is a parameter of the first Poisson process;
 *         lambda2 is a parameter of the second Poisson process;
 *         count_new_type - the maximum number of changes.
 *      
 *     OUTPUT: dimension vector (count_iter + 1) with elements 0, 1 and 2 (the first element is 0):
 *         0 - Do not change anything;
 *         1 - Replace the existing view with a new one;
 *         2 - Add a new one.                                                                                                                                                 */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_vector *poisson_new_element(int count_iter, double lambda1, double lambda2, int count_new_type)
{
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	
	double a = 0, b = 1;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	gsl_vector_set(x, 0, 0);
	
	int cnt = 0; 
    for(int i = 1; i <= count_iter; i++)
    {
		if (gsl_ran_flat(r, a, b) <= lambda1 and cnt < count_new_type) 
		{
			cnt++;
			if (gsl_ran_flat(r, a, b) <= lambda2)  gsl_vector_set(x, i, 2);
			else gsl_vector_set(x, i, 1);
		}
		else gsl_vector_set(x, i, 0);
	}
	
	gsl_rng_free(r);
	
	return x;
}



/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
 * If a new element is added to the system, it is necessary to recalculate the interaction matrix and the fixed point. 
 * In this case, for the corresponding iterations, we generate a random beta value from a uniform distribution
 *                    
 *     INPUT: 
 *         count_iter - the number of iterations;
 *         poisson_vector - vector obtained above.
 *      
 *     OUTPUT: dimension vector (count_iter + 1) with elements 0 and beta:
 *         0 - if there were no changes or the existing element was changed;
 *         beta - if a new element is added to the system.                                                                                                                                                 */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
gsl_vector *get_beta_vector(int count_iter, gsl_vector *poisson_vector)
{
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	
	double a = 0.5, b = 0.999;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	
    for(int i = 0; i <= count_iter; i++)
    {
		if (gsl_vector_get(poisson_vector, i) == 2) 
			gsl_vector_set(x, i, gsl_ran_flat(r, a, b));
		else gsl_vector_set(x, i, 0);
		gsl_ran_flat(r, a, b);
	}
	
	gsl_rng_free(r);
	
	return x;
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*Определяем на каких итерациях будем решать ОДУ:
                   
                    1. Если на текущем или следующем шаге будет добавляться новый вид
                    2. Если текущий шаг кратен заданному шагу решения (solve step)
                    3. Если теущий шаг первый (нулевой, т.е. когда системы исходная)                                                                                          */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_vector *get_step_odu(int count_iter, int solve_step, gsl_vector *poisson_vector)
{
	int j = 0;
	gsl_vector *x = gsl_vector_alloc(count_iter + 1);
	gsl_vector_set(x, 0, 1);
	
	for(int i = 1; i <= count_iter; i++)
	{
		if(gsl_vector_get(poisson_vector, i) > 0)
		{
			gsl_vector_set(x, i, 1);
			gsl_vector_set(x, i - 1, 1);
		}
		else gsl_vector_set(x, i, 0); 
		
		if(j == solve_step)
		{
			gsl_vector_set(x, i, 1);
			j = -1;
		} 
		
		j++;
	}
	
	return x;
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*Solve SLAE: find the coordinates of a fixed point
  *
  *     INPUT:
  *         sizeA - the size of the interaction matrix A
  *         A - interaction matrix
  *
  *     OUTPUT:
  *         x - coordinates of a fixed point
 */                                                                               
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_vector *get_freq(int sizeA, gsl_matrix *A)
{
    gsl_matrix *left_part = gsl_matrix_alloc(sizeA, sizeA);

    gsl_vector *v1 = gsl_vector_alloc(sizeA);
    gsl_vector *v2 = gsl_vector_alloc(sizeA);

    for(int j = 1; j < sizeA; j++)
    {
        gsl_matrix_get_row(v1, A, 0);
        gsl_matrix_get_row(v2, A, j);
        gsl_vector_sub(v1, v2);
        gsl_matrix_set_row(left_part, (j - 1), v1);            
    }
    gsl_vector_set_all(v1, 1);
    gsl_matrix_set_row(left_part, (sizeA - 1), v1);
    gsl_vector_free(v1);
    gsl_vector_free(v2);

    gsl_vector *right_part = gsl_vector_calloc(sizeA);
    gsl_vector_set(right_part, (sizeA - 1), 1);

    gsl_vector *x = gsl_vector_alloc(sizeA);
    
    int s;
    gsl_permutation *p = gsl_permutation_alloc(sizeA);
    gsl_linalg_LU_decomp(left_part, p, &s);
    gsl_linalg_LU_solve(left_part, p, right_part, x);
    
    gsl_vector_free(right_part);
    gsl_matrix_free(left_part);
    gsl_permutation_free(p);

    return x;    
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Find eigenvalues:
 * The results of this function are not recorded anywhere.
 * If necessary, they can be viewed in the command window
 */                                                                           
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

void get_eigen_value(gsl_matrix *A, gsl_vector *x, int sizeA)
{
	double data[sizeA * sizeA];
	
	for(int i = 0; i < sizeA; i++)
    {
		for(int j = 0; j < sizeA; j++)
		{
			data[i * sizeA +  j] = gsl_vector_get(x, i) * gsl_matrix_get(A, i, j);
			for(int k = 0; k < sizeA; k++)
			    data[i * sizeA + j] = data[i * sizeA + j] - gsl_vector_get(x, i) * gsl_vector_get(x, k) * (gsl_matrix_get(A, k, j) + gsl_matrix_get(A, j, k));
			
			if (i == j)
			{
				for(int k = 0; k < sizeA; k++)
				{
					data[i * sizeA + i] = data[i * sizeA + i] + gsl_vector_get(x, k) * gsl_matrix_get(A, i, k);
				    for(int l = 0; l < sizeA; l++)
				       data[i * sizeA + i] = data[i * sizeA + i] - gsl_vector_get(x, k) * gsl_vector_get(x, l) * gsl_matrix_get(A, k, l);
				} 
			} 
		}
	}
	
	gsl_matrix_view m = gsl_matrix_view_array (data, sizeA, sizeA);
    gsl_vector_complex *eval = gsl_vector_complex_alloc (sizeA);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc (sizeA, sizeA);
    gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc (sizeA);
    gsl_eigen_nonsymmv (&m.matrix, eval, evec, w);
    gsl_eigen_nonsymmv_free (w);
    gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);
    
    {
		int i, j;
		for (i = 0; i < sizeA; i++)
		{
			gsl_complex eval_i = gsl_vector_complex_get (eval, i);
			gsl_vector_complex_view evec_i = gsl_matrix_complex_column (evec, i);
			printf ("eigenvalue = %g + %gi\n", GSL_REAL(eval_i), GSL_IMAG(eval_i));
			printf ("eigenvector = \n");
			for (j = 0; j < sizeA; ++j)
			{
				gsl_complex z = gsl_vector_complex_get(&evec_i.vector, j);
				printf("%g + %gi\n", GSL_REAL(z), GSL_IMAG(z));
			}
		}
	}
	gsl_vector_complex_free(eval);
	gsl_matrix_complex_free(evec);
	
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                                                         Solve ODE                                                                                          */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

int func (double t, const double y[], double f[], void *params)
{
    (void)(t); 
    gsl_matrix *A = (gsl_matrix*)params;
        
    for(int i = 0; i < A->size1; i++)
    {
		f[i] = 0;
		for(int j = 0; j < A->size1; j++)
		{
			f[i] = f[i] + gsl_matrix_get(A, i, j) * y[j];
			for(int k = 0; k < A->size1; k++)
			{
				f[i] = f[i] - gsl_matrix_get(A, k, j) * y[k] * y[j];
			}
		}
		f[i] = f[i] * y[i];
	}
    
    return GSL_SUCCESS;
}

int jac (double t, const double y[], double *dfdy, double dfdt[], void *params)
{
    (void)(t); 
    gsl_matrix *A = (gsl_matrix*)params;
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, A->size1, A->size1);
    gsl_matrix * m = &dfdy_mat.matrix;
    
    for(int i = 0; i < A->size1; i++)
    {
		for(int j = 0; j < A->size1; j++)
		{
			gsl_matrix_set(m, i, j, y[i] * gsl_matrix_get(A, i, j));
			for(int k = 0; k < A->size1; k++)
			    gsl_matrix_set(m, i, j, gsl_matrix_get(m, i, j) - y[i] * y[k] * (gsl_matrix_get(A, k, j) + gsl_matrix_get(A, j, k)));
			
			if (i == j)
			{
				for(int k = 0; k < A->size1; k++)
				{
					gsl_matrix_set(m, i, i, gsl_matrix_get(m, i, i) + y[k] * gsl_matrix_get(A, i, k));
				    for(int l = 0; l < A->size1; l++)
				       gsl_matrix_set(m, i, i, gsl_matrix_get(m, i, i) - y[k] * y[l] * gsl_matrix_get(A, k, l));
				} 
			} 
		}
	}
    
    for(int i = 0; i < A->size1; i++)
        dfdt[i] = 0.0;
 
    return GSL_SUCCESS;
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                                       Calculate average integral fitness (quadrature formulas)                                                             */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

double get_avg_integral_fitness(gsl_vector *U_continuos, gsl_matrix *A, int sizeA, double count_step, int sizeU_)
{
	double s, f = 0;
	int start = sizeU_ - sizeA * (count_step + 1);
	for(int i = 0; i <= count_step; i++)
	{
		s = 0;
		for(int j = 0; j < sizeA; j++)
		    for(int k = 0; k < sizeA; k++)
		        s = s + gsl_matrix_get(A, j, k) * gsl_vector_get(U_continuos, start + i * sizeA + j) * gsl_vector_get(U_continuos, start + i * sizeA + k);
		
		if ((i == 0) || (i == count_step)) s = s / 2;
		f = f + s;
    }
	   
	f = f / count_step;
	
	return f;
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* Solve the linear programming problem: find the increments of the interaction matrix
  *
  *    INPUT:
  *        constr - restriction on bij elements
  *        A - interaction matrix
  *        x - coordinates of a fixed point
  *        sizeA - the size of the interaction matrix
  *
  *    OUTPUT:
  *        B - matrix of increments of the elements of the interaction matrix
 */                                                                
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_matrix *solve_lin_prog(double constr, gsl_matrix *A, gsl_vector *x, int sizeA)
{
    /* Find the inverse matrix to the matrix A */
    gsl_matrix *invA = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix   *A2 = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix_memcpy(A2, A);
    int s;

    gsl_permutation *p = gsl_permutation_alloc(sizeA);
    gsl_linalg_LU_decomp(A2, p, &s);
    gsl_linalg_LU_invert(A2, p, invA);
    gsl_matrix_free(A2);
    gsl_permutation_free(p);

    /* Find the coefficients before the elements bij */
    gsl_matrix *B = gsl_matrix_alloc(sizeA, sizeA);
    gsl_matrix_set_zero(B);
    double a;

    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            for(int k = 0; k < sizeA; k++)
            {
                a = 0;
                for(int m = 0; m < sizeA; m++)
                {
                    a = a + gsl_matrix_get(invA, j, m);
                }
                gsl_matrix_set(B, i, j, gsl_matrix_get(B, i, j) + a * gsl_matrix_get(invA, k, i));
            }
        }
    } 

    /* We set the linear programming problem */
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MAX);

    /* Restrictions on the sum of bij and on the sum of aij * bij
     * In addition, if there are fixed point coordinates tending to 0 or 1,
     * impose restrictions on the increments of the corresponding coordinates
    */
    int count_chng = 0, count_chng2 = 2;
    for(int i = 0; i < sizeA; i++)
        if((gsl_vector_get(x, i) <= EPS) || (gsl_vector_get(x, i) >= (1 - EPS))) count_chng++;
    
    glp_add_rows(lp, 1 + count_chng);
    glp_set_row_bnds(lp, 1, GLP_UP, 0.0, 0.0);
    
    if(count_chng > 0)
    { 
		for(int i = 0; i < sizeA; i++)
		{
			if(gsl_vector_get(x, i) <= EPS)
			{
				glp_set_row_bnds(lp, count_chng2, GLP_LO, 0, 0);
				count_chng2++;
			}
			
			if(gsl_vector_get(x, i) >= (1 - EPS))
			{
				glp_set_row_bnds(lp, count_chng2, GLP_UP, 0, 0);
				count_chng2++;
			}
			
			if((count_chng2 - 2) >= count_chng) break;
		}
    }

    int ia[(count_chng + 1) * sizeA * sizeA + 1], ja[(count_chng + 1) * sizeA * sizeA + 1];
    double ar[(count_chng + 1) * sizeA * sizeA + 1];
    int ind1 = 1;
    count_chng2 = 1;

    for(int k = 1; k <= (sizeA + 1); k++)
    {
		if((k == 1) || (gsl_vector_get(x, k - 2) <= EPS) || (gsl_vector_get(x, k - 2) >= (1 - EPS)))
		{
			for(int i = 0; i < sizeA; i++)
			{
				for(int j = 0; j < sizeA; j++)
				{
					ia[ind1] = count_chng2;
					ja[ind1] = i * sizeA + j + 1;
					
					if (count_chng2 == 1) 
						ar[ind1] = gsl_matrix_get(A, i, j);
					else
					{
						ar[ind1] = 0;
						for(int m = 0; m < sizeA; m++)
						{
							ar[ind1] = ar[ind1] + gsl_matrix_get(invA, m, i) * gsl_vector_get(x, j) * gsl_vector_get(x, k - 2);
						}
						ar[ind1] = ar[ind1] - gsl_matrix_get(invA, k - 2, i) * gsl_vector_get(x, j);
					}
					
					ind1 = ind1 + 1;        
				}
			}
			count_chng2++;
	    }
	}
	gsl_matrix_free(invA);

    /* Restrictions on each bij left and right */
    ind1 = 1;
    glp_add_cols(lp, sizeA * sizeA);
    for(int i = 0; i < sizeA * sizeA; i++)
    {
        glp_set_col_bnds(lp, ind1, GLP_DB, -constr, constr);
        ind1 = ind1 + 1;  
    }

    /* The coefficients in the equation that we want to maximize */
    ind1 = 1;
    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            glp_set_obj_coef(lp, ind1, gsl_matrix_get(B, i, j));
            ind1 = ind1 + 1;        
        }
    }

    /* Solve the linear programming problem */
    glp_load_matrix(lp, (count_chng + 1) * sizeA * sizeA, ia, ja, ar);
    glp_simplex(lp, NULL);

    ind1 = 1;
    for(int i = 0; i < sizeA; i++)
    {
        for(int j = 0; j < sizeA; j++)
        {
            gsl_matrix_set(B, i, j, glp_get_col_prim(lp, ind1));
            ind1 = ind1 + 1;        
        }
    }

    glp_delete_prob(lp);

    return B;
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                                                 Recalculate the interaction matrix                                                                         */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

gsl_matrix *get_new_matrix_A(gsl_matrix *A, int sizeA, gsl_vector *x, int flg, gsl_vector *num_view, int size_vec, double constr, double old_f, double beta)
{
	gsl_matrix *newA = gsl_matrix_alloc(sizeA, sizeA);
	
	if(flg == 1)
	{
		double a, b;
		const gsl_rng_type * T;
		gsl_rng * r;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc(T);
	
		int num_min_type = gsl_vector_min_index(x);
		gsl_vector_set(num_view, size_vec, num_min_type + 1);
        gsl_matrix_memcpy(newA, A);
        
        for(int i = 0; i < sizeA; i++)
        {	      
		    a = gsl_matrix_get(A, num_min_type, i);
		    b = 1.005 * a;    
		    gsl_matrix_set(newA, num_min_type, i, gsl_ran_flat(r, a, b));
		}
		gsl_rng_free(r);				
	}
	else
	{
		gsl_vector_set(num_view, size_vec, sizeA);
		
        double new_f = 0;
        
        /* Solve the linear programming problem */
        gsl_matrix *B = gsl_matrix_alloc(sizeA - 1, sizeA - 1); 
        gsl_matrix *A_copy = gsl_matrix_alloc(sizeA - 1, sizeA - 1);
        B = solve_lin_prog(constr, A, x, sizeA - 1);
        gsl_matrix_memcpy(A_copy, A);
        
        /* Rewrite the matrix A */
        gsl_matrix_add(A_copy, B);
        gsl_matrix_free(B);
        
        /* Find a fixed point */
        gsl_vector *y = gsl_vector_alloc(sizeA - 1);
        y = get_freq(sizeA - 1, A_copy);
        
        /* Calculate Fitness */
        for(int i = 0; i < sizeA - 1; i++)
            for(int j = 0; j < sizeA - 1; j++)
                new_f = new_f + gsl_matrix_get(A_copy, i, j) * gsl_vector_get(y, i) * gsl_vector_get(y, j);
        gsl_matrix_free(A_copy);        
		gsl_vector_free(y);
		
		for(int i = 0; i < sizeA - 1; i++)
		    for(int j = 0; j < sizeA - 1; j++)
		        gsl_matrix_set(newA, i, j, gsl_matrix_get(A, i, j));
		      
		double alpha = 1 / (gsl_vector_max(x) + 1) * beta;
		     
        for(int i = 0; i < sizeA - 1; i++)
        {
            gsl_matrix_set(newA, i, sizeA - 1, (new_f - alpha * old_f) / (1 - alpha));
            gsl_matrix_set(newA, sizeA - 1, i, new_f / (alpha * sizeA * gsl_vector_get(x, i)));
		}
		gsl_matrix_set(newA, sizeA - 1, sizeA - 1, new_f / ((1 - alpha) * sizeA));
		
		
	}
	
	return newA;
}



/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*    Write data to a file just to see it with your eyes:
 *        write to the file the evolution of the interaction matrix, 
 *        the fitness system,
 *        the average integral fitness,
 *        the matrix norm at different iterations 
 */                                                                         
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

void write_in_file(gsl_vector *A_time, int start_sizeA, int count_iter, gsl_vector * poisson_vector, 
                   gsl_vector *fitness_vec, gsl_vector *fitness_vec_avg, int count_solve_step, gsl_vector *matrix_norm_vec)
{
	
	/* Write the type of matrix A at each step */
	int ind = 0;
	ofstream evolution_A("evolution_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
	{
		if(gsl_vector_get(poisson_vector, i) == 2) start_sizeA++;
		for(int j = 0; j < start_sizeA; j++)
		{
		    for(int k = 0; k < start_sizeA; k++)
		    {
				evolution_A << gsl_vector_get(A_time, ind) << " ";
				ind++;
			}	
			evolution_A << endl;
		}
		evolution_A << endl;
	}
	evolution_A.close();
	
	
	/* Write fitness at every step */
	ofstream fitness("fitness.txt");
	for(int i = 0; i <= count_iter; i++)
		fitness << gsl_vector_get(fitness_vec, i) << endl;
	fitness.close();
	
	/* Write average integral fitness */
	ofstream fitness_avg("fitness_avg.txt");
	for(int i = 0; i < count_solve_step; i++)
		fitness_avg << gsl_vector_get(fitness_vec_avg, i) << endl;
	fitness.close();
	
	/* Write the norm of the matrix A at each step */
	ofstream norm_A("norma_matrix_A.txt");
	for(int i = 0; i <= count_iter; i++)
		norm_A << gsl_vector_get(matrix_norm_vec, i) << endl;
	norm_A.close();
}



/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                                  Write the data to a file in a binary form, then we consider it as Matlab                                                  */
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
void write_in_file_for_Matlab(gsl_vector * poisson_vector, int count_iter, gsl_vector *num_view, int size_vec,
                              gsl_vector *fitness_vec, gsl_vector *fitness_vec_avg, int count_solve_step, gsl_vector *solve_odu_vector,
                              gsl_vector *U, int sizeU2, int sizeA, gsl_vector *count_view, gsl_vector *U_continuos, int sizeU, gsl_vector *time_vec, int count_step,
                              gsl_vector *beta_vector) 
{
	double num;
	int num2;
	
	/* Write custom data for MatLab */
	ofstream set("settings_matlab.txt");
	set.write((char*)&count_iter, sizeof count_iter); /*Number of iterations*/
	set.write((char*)&size_vec, sizeof size_vec); /* Number of additions / replacements */
	set.write((char*)&count_solve_step, sizeof count_solve_step); /* Number of iterations at which an ODE is solved */
	set.write((char*)&sizeA, sizeof sizeA); /* Number of species at the last moment */
	set.write((char*)&count_step, sizeof count_step); /* The number of steps to solve a continuous problem */
	set.close();

	
	/* Write the results of the Poisson process */
	ofstream poisson("poisson_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num2 = gsl_vector_get(poisson_vector, i); 
		poisson.write((char*)&num2, sizeof num2);
	}
	poisson.close();
	
	/*Write the beta vector*/
	ofstream beta("beta_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num = gsl_vector_get(beta_vector, i); 
		beta.write((char*)&num, sizeof num);
	}
	beta.close();
	
	/* Write numbers of new or replaceable types */
	ofstream view("num_view_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i < size_vec; i++)
    {
	    num2 = gsl_vector_get(num_view, i); 
		view.write((char*)&num2, sizeof num2);
	}
	view.close();
	
	/* Write the fitness vector */
	ofstream fitn("fitness_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_iter; i++)
	{
		num = gsl_vector_get(fitness_vec, i);
		fitn.write((char*)&num, sizeof num);
	}				
	fitn.close();
	
	/* Write the vector of average integral fitness */
	ofstream fitn_avg("fitness_avg_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < count_solve_step; i++)
	{
		num = gsl_vector_get(fitness_vec_avg, i);
		fitn_avg.write((char*)&num, sizeof num);
	}				
	fitn_avg.close();
	
	/*iterations on which we solve ODE*/
	ofstream solve_odu("solve_odu_matlab.txt", ios::binary | ios::out);
	
	for(int i = 0; i <= count_iter; i++)
    {
	    num2 = gsl_vector_get(solve_odu_vector, i); 
		solve_odu.write((char*)&num2, sizeof num2);
	}
	solve_odu.close();
	
	/* Write frequencies */
	ofstream freq("freqType_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeU2; i++)
	{
		num = gsl_vector_get(U, i); 
		freq.write((char*)&num, sizeof num);
	}
	freq.close();
	
	/* Write the number of species at each iteration */
	ofstream view2("count_view_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_iter; i++)
	{
		num2 = gsl_vector_get(count_view, i); 
		view2.write((char*)&num2, sizeof num2);
	}
	view2.close();
	
    /* Write the decisions of the ODE */
	ofstream freq_cont("freqType_continuos_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i < sizeU; i++)
	{
	    num = gsl_vector_get(U_continuos, i);
		freq_cont.write((char*)&num, sizeof num);
	}
	freq_cont.close();
	
	/* Write the time vector */
	ofstream time("time_matlab.txt", ios::binary | ios::out);
	for(int i = 0; i <= count_step; i++)
	{
		num = gsl_vector_get(time_vec, i);
		time.write((char*)&num, sizeof num);
	} 
	time.close();
}


int main(int *argc, char **argv)
{
    /* Enter the data from the keyboard:
    *     interaction matrix size
    *     the number of iterations of evolution
    *     the parameters of Poisson processes 
    *     the final point in time to solve the ODE
    *     time step (grid for solving ODE)
    *     ODU decision step (at which iterations of evolution we will solve an ODE) 
    *     the maximum number of changes
    */
    int sizeA, count_iter, solve_step, count_new_type;
    double t1, h, lambda1, lambda2;
    
    /*interaction matrix size*/
    cout << "Enter the size of the matrix A "; cin >> sizeA; cout << endl; int start_sizeA = sizeA;
    /*the number of iterations of evolution*/
    cout << "Enter the count iteration "; cin >> count_iter; cout << endl;
    /*lambda1 - to make changes or not*/
    cout << "Enter the Lambda1 "; cin >> lambda1; cout << endl;
    /*lambda2 - in case of making changes first, determines to add a new view or replace an existing one*/
    cout << "Enter the Lambda2 "; cin >> lambda2; cout << endl;
    /*the final point in time to solve the ODE*/
    cout << "Enter T1 "; cin >> t1; cout << endl;
    /*time step (grid for solving ODE)*/
    cout << "Enter the time step "; cin >> h; cout << endl;   
    /*ODU decision step (at which iterations of evolution we will solve an ODE) */
    cout << "Enter the solve dif.eq. step "; cin >> solve_step; cout << endl;   
    /*the maximum number of changes*/
    cout << "Enter the maximum number of changes "; cin >> count_new_type; cout << endl;

    /*We read from the files the interaction matrix, the restrictions on the bij elements, as well as the initial conditions*/
    gsl_matrix *A = gsl_matrix_alloc(sizeA, sizeA);
    gsl_vector *u0 = gsl_vector_alloc(sizeA);
    double constr;
    
    double buff;
    ifstream fin_A("Matrix_A.txt");// opened the file for reading the interaction matrix
    ifstream fin_B_constr("Matrix_B_constr.txt");// opened the file to read the restrictions on the elements bij
    ifstream fin_u0("u0.txt");// opened the file to read the initial data
    for(int i = 0; i < sizeA; i++)
    {
		fin_u0 >> buff; gsl_vector_set(u0, i, buff);
        for(int j = 0; j < sizeA; j++)
        {
            fin_A >> buff;   
            gsl_matrix_set(A, i, j, buff);         
        }
    }        
    fin_B_constr >> buff;
    constr = buff;
    fin_A.close(); fin_B_constr.close(); fin_u0.close();// close the file
    
    /*We determine at what iterations we will add a new one or replace an existing one*/
    gsl_vector *poisson_vector;
    poisson_vector = poisson_new_element(count_iter, lambda1, lambda2, count_new_type);
    
    /* If a new element is added to the system, it is necessary to recalculate the interaction matrix and the fixed point. 
     * In this case, for the corresponding iterations, we generate a random beta value from a uniform distribution
    */
    gsl_vector *beta_vector;
    beta_vector = get_beta_vector(count_iter, poisson_vector);
    
    /*we define at what iterations we will solve ODE*/
    gsl_vector *solve_odu_vector;
    solve_odu_vector = get_step_odu(count_iter, solve_step, poisson_vector);
    
    /*calculate the number of steps to solve an ODE and determine the parameters for its solution*/
    int count_solve_step = 0, sizeA_max = sizeA, sizeU = 0, sizeU2 = 0, sizeA2 = 0;
    double count_step, t0;
    modf(t1 / h, &count_step);
    
    for(int i = 0; i <= count_iter; i++)
    {
		if(gsl_vector_get(solve_odu_vector, i) == 1)
		{ 
			count_solve_step++;
			if(gsl_vector_get(poisson_vector, i) == 2) sizeA_max++;
			sizeU = sizeU + sizeA_max * (count_step + 1); 
		} 
		sizeU2 = sizeU2 + sizeA_max;
		sizeA2 = sizeA2 + sizeA_max * sizeA_max;
	}
	gsl_vector *U_continuos = gsl_vector_alloc(sizeU);
	
	gsl_odeiv2_driver * d;
    gsl_odeiv2_system sys;
    
    gsl_vector *time_vec = gsl_vector_alloc(count_step + 1); 
    for(int i = 0; i <= count_step; i++)
        gsl_vector_set(time_vec, i, h * i);
	
    /*Output*/
    //equilibrium matrix at each iteration
    gsl_vector *U = gsl_vector_alloc(sizeU2);
    //average fitness vector at each iteration
    gsl_vector *fitness_vec = gsl_vector_alloc(count_iter + 1);
    //The vector of average integral fitness
    gsl_vector *fitness_vec_avg = gsl_vector_alloc(count_solve_step);
    //The vector of evolutionary changes in the elements of the interaction matrix
    gsl_vector *A_time = gsl_vector_alloc(sizeA2); 
    //matrix norm at each iteration
    gsl_vector *matrix_norm_vec = gsl_vector_alloc(count_iter + 1);
  
    //Vector in which we will write the number of the new species or the number of the species that will be replaced 
    gsl_vector *num_view;
    int size_vec = 0;
    for(int i = 0; i <= count_iter; i++)
        if(gsl_vector_get(poisson_vector, i) > 0) size_vec++;   
        
    if(size_vec == 0) size_vec++;    
    num_view = gsl_vector_alloc(size_vec);
    size_vec = 0;
    
    /* Vector with the number of species at each iteration */
    gsl_vector *count_view = gsl_vector_alloc(count_iter + 1);
        
    /* Vector to record intermediate fixed point calculations */
    gsl_vector *x;    
    /* Matrix for recording intermediate calculations of the LPP solution */
    gsl_matrix *B;  
     
    int sizeU2_ = 0, sizeA2_ = 0, sizeU_ = 0, count_solve_step2 = 0;
    
    /* At each next step, we again find the frequencies and fitness, solve the LPP and get a new matrix A */
    for(int i = 0; i <= count_iter; i++)
    {
		cout << "I = " << i << endl << endl; /* Print the iteration number */
		    
        /* Find a fixed point */
        x = get_freq(sizeA, A);
        
        /* Find eigenvalues */
        get_eigen_value(A, x, sizeA);
        
        if(gsl_vector_min(x) >= 0)
        {
			/* Write the number of species */
			gsl_vector_set(count_view, i, sizeA);
			
			/* Write the equilibrium position */
			for(int k = 0; k < sizeA; k++)
			{
				gsl_vector_set(U, sizeU2_, gsl_vector_get(x, k));
				sizeU2_++;
			}  

            /* Calculate Fitness */
			gsl_vector_set(fitness_vec, i, 0);
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
					gsl_vector_set(fitness_vec, i, gsl_vector_get(fitness_vec, i) + gsl_matrix_get(A, k, j) * gsl_vector_get(x, k) * gsl_vector_get(x, j));
					
			/* Calculate the matrix norm */
			gsl_vector_set(matrix_norm_vec, i, 0);
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
				    gsl_vector_set(matrix_norm_vec, i, gsl_vector_get(matrix_norm_vec, i) + gsl_matrix_get(A, k, j) * gsl_matrix_get(A, k, j));
	 
			/* Save the values of the elements of the interaction matrix */
			for(int k = 0; k < sizeA; k++)
				for(int j = 0; j < sizeA; j++)
				{
					gsl_vector_set(A_time, sizeA2_, gsl_matrix_get(A, k, j));
					sizeA2_++;
				}
					
			/* Solve the ODE and write the result in the matrix */
			if (gsl_vector_get(solve_odu_vector, i) == 1)
			{
				double y[sizeA];
				sys = {func, jac, sizeA, A};
				d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);
			
				for(int k = 0; k < sizeA; k++) 
				{
					y[k] = gsl_vector_get(u0, k);
					gsl_vector_set(U_continuos, sizeU_, y[k]);
					sizeU_++;
				}
			
				t0 = 0.0;
				for(int k = 0; k < count_step; k++)
				{
					double ti = h * (k + 1);
					int status = gsl_odeiv2_driver_apply (d, &t0, ti, y);
					if (status != GSL_SUCCESS)
					{
						printf ("error, return value=%d\n", status);
						break;
					}
					for(int j = 0; j < sizeA; j++){
					    gsl_vector_set(U_continuos, sizeU_, y[j]); 
					    sizeU_++;
					}
				}
				
				/* Calculate average integral fitness */
				gsl_vector_set(fitness_vec_avg, count_solve_step2, get_avg_integral_fitness(U_continuos, A, sizeA, count_step, sizeU_));
				
				gsl_odeiv2_driver_free(d);  
				count_solve_step2++;
			}      
	    }
	    
	    else
	    {
		    cout << "COMPONENTS LESS 0" << endl;
		    
		    for(int l = 0; l < sizeA; l++) cout << gsl_vector_get(x, l) << " ";
		    cout << endl << endl;
		    
		    for(int l = 0; l <= i; l++) cout << gsl_vector_get(poisson_vector, l) << " ";
		    cout << endl;
		    break;    	
		}
		 
		if(i < count_iter)
		{      
			/* Check whether you need to enter a new element */
			if(gsl_vector_get(poisson_vector, (i + 1)) > 0)
			{
				/* Rewrite the matrix A: The case of changing an existing element */
				if(gsl_vector_get(poisson_vector, (i + 1)) == 1)
				{
				    A = get_new_matrix_A(A, sizeA, x, 1, num_view, size_vec, constr, gsl_vector_get(fitness_vec, i), gsl_vector_get(beta_vector, i + 1));
				    x = get_freq(sizeA, A);
				}
				else
				{				
					
					sizeA++;
					/* Rewrite the matrix A: The case of adding a new  element */
					gsl_matrix *newA = gsl_matrix_alloc(sizeA, sizeA);
					newA = get_new_matrix_A(A, sizeA, x, 2, num_view, size_vec, constr, gsl_vector_get(fitness_vec, i), gsl_vector_get(beta_vector, i + 1));					
					gsl_matrix_free(A);
					A = gsl_matrix_alloc(sizeA, sizeA);
					gsl_matrix_memcpy(A, newA);
					gsl_matrix_free(newA);
					
					/* Rewrite fixed point */
					double alpha = 1 / (gsl_vector_max(x) + 1) * gsl_vector_get(beta_vector, i + 1); //fixed point coordinate corresponding to new element
					
					gsl_vector *newX = gsl_vector_alloc(sizeA);
					for(int k = 0; k < sizeA - 1; k++)
					    gsl_vector_set(newX, k, gsl_vector_get(x, k) * alpha);
					gsl_vector_set(newX, sizeA - 1, (1 - alpha));
					
					gsl_vector_free(x);
					x = gsl_vector_alloc(sizeA);
					gsl_vector_memcpy(x, newX);
					gsl_vector_free(newX);
					
					/*Rewrite initial data*/
					gsl_vector *new_u0 = gsl_vector_alloc(sizeA);
					gsl_vector_set(new_u0, (sizeA - 1), 0);
					for(int k = 0; k < (sizeA - 1); k++)
						gsl_vector_set(new_u0, k, 0.9 * gsl_vector_get(u0, k));
					gsl_vector_set(new_u0, (sizeA - 1), 0.1);
					
					gsl_vector_free(u0);
					u0 = gsl_vector_alloc(sizeA);
					gsl_vector_memcpy(u0, new_u0);
					gsl_vector_free(new_u0);
				}
				size_vec++;			
			}
	    } 		
		
		if(i < count_iter)
		{ 
			if(gsl_vector_get(poisson_vector, (i + 1)) == 0)
			{
				/* Solve the linear programming problem */
				B = solve_lin_prog(constr, A, x, sizeA);
				/* Rewrite the matrix A */
				gsl_matrix_add(A, B);
				
				gsl_matrix_free(B);
			}
	    }
        gsl_vector_free(x);
    }   
    
    write_in_file(A_time, start_sizeA, count_iter, poisson_vector, fitness_vec, fitness_vec_avg, count_solve_step, matrix_norm_vec);
    write_in_file_for_Matlab(poisson_vector, count_iter, num_view, size_vec, fitness_vec, fitness_vec_avg, 
                             count_solve_step, solve_odu_vector, U, sizeU2, sizeA, count_view, U_continuos, sizeU, time_vec, count_step, beta_vector);
    
    gsl_matrix_free(A);
    gsl_vector_free(u0);
    gsl_vector_free(poisson_vector);
    gsl_vector_free(beta_vector);
    gsl_vector_free(solve_odu_vector);
    gsl_vector_free(U_continuos);
    gsl_vector_free(time_vec); 
    gsl_vector_free(U);
    gsl_vector_free(fitness_vec);
    gsl_vector_free(fitness_vec_avg);
    gsl_vector_free(A_time);
    gsl_vector_free(matrix_norm_vec);
    gsl_vector_free(num_view);
    gsl_vector_free(count_view);
            
    return 0;
}



