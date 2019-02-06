import sys
import random
import time
from . import oph

#fitness will take a set s and a set of weights and return a tuple containing the fitness and the best path
def fitness( chrom, s, start_point, end_point, tmax ):
    augs = []
    for i in range( len( s ) ):
        augs.append( ( s[ i ][0],
                       s[ i ][1],
                       s[ i ][2],
                       s[ i ][3], 
                       s[ i ][4] + chrom[ i ] ) )
    if debug:
        print ('fitness---------------------------------')
        print ('augs:')
        print (augs)
    #best = oph.ellinit_replacement( augs, start_point, end_point, tmax )
    ellset = oph.ell_sub( tmax, start_point, end_point, augs )
    #best = oph.initialize( ellset, start_point, end_point, tmax )[0]
    best = oph.init_replacement( ellset, start_point, end_point, tmax )[0]
    if debug:
        print ('best:')
        print (best)
        print ('best real reward:')
        print ([ x[3] for x in best ])
        print (len( s ))
        print ([ s[ x[3] - 2 ] for x in best[ 1:len( best ) - 1 ] ])
        print ([ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ])
        print (( sum( [ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ] ), best ))
    return ( sum( [ s[ x[3] - 2 ][2] for x in best[ 1:len( best ) - 1 ] ] ), best )

def crossover( c1, c2 ):
    assert( len( c1 ) == len( c2 ) )
    point = random.randrange( len( c1 ) )
    first = random.randrange( 2 )
    if( first ):
        return c1[:point] + c2[point:]
    else:
        return c2[:point] + c1[point:]

def mutate( chrom, mchance, msigma ):
    return [ x + random.gauss( 0, msigma ) if random.randrange( mchance ) == 0  else 
             x for x in chrom ]

def run_alg_f( f, tmax, N ):
    random.seed()
    cpoints = []
    an_unused_value = f.readline() # ignore first line of file
    for i in range( N ):
        cpoints.append( tuple( [ float( x ) for x in f.readline().split() ] ) )
    if debug:
        print ('N:            ', N)
    return run_alg(cpoints, tmax)

def run_alg(points, tmax, return_sol=False, verbose=True):
    cpoints = [tuple(p) + (i, 0) for i, p in enumerate(points)]
    start_point = cpoints.pop( 0 )
    end_point = cpoints.pop( 0 )
    assert( oph.distance( start_point, end_point ) < tmax )
    popsize = 10
    genlimit = 10
    kt = 5
    isigma = 10
    msigma = 7
    mchance = 2
    elitismn = 2
    if( debug ):
        print ('data set size:', len( cpoints ) + 2)
        print ('tmax:         ', tmax)
        print ('parameters:')
        print ('generations:     ', genlimit)
        print ('population size: ', popsize)
        print ('ktournament size:', kt)
        print ('mutation chance: ', mchance)
        print (str( elitismn ) + '-elitism')

    start_time = time.clock()
    #generate initial random population
    pop = []
    for i in range( popsize + elitismn ):
        chrom = []
        for j in range( len( cpoints ) ):
            chrom.append( random.gauss( 0, isigma ) )
        chrom = ( fitness( chrom, cpoints, start_point, end_point, tmax )[0], chrom )
        while( i - j > 0 and j < elitismn and chrom > pop[ i - 1 - j ] ):
            j += 1
        pop.insert( i - j, chrom )

    bestfit = 0
    for i in range( genlimit ):
        nextgen = []
        for j in range( popsize ):
            #select parents in k tournaments
            parents = sorted( random.sample( pop, kt ) )[ kt - 2: ] #optimize later
            #crossover and mutate
            offspring = mutate( crossover( parents[0][1], parents[1][1] ), mchance, msigma )
            offspring = ( fitness( offspring, cpoints, start_point, end_point, tmax )[0], offspring )
            if( offspring[0] > bestfit ):
                bestfit = offspring[0]
                if verbose:
                    print (bestfit)
            if( elitismn > 0 and offspring > pop[ popsize ] ):
                l = 0
                while( l < elitismn and offspring > pop[ popsize + l ] ):
                    l += 1
                pop.insert( popsize + l, offspring )
                nextgen.append( pop.pop( popsize ) )
            else:
                nextgen.append( offspring )
        pop = nextgen + pop[ popsize: ]

    bestchrom = sorted( pop )[ popsize + elitismn - 1 ] 
    end_time = time.clock()

    if verbose:
        print ('time:')
        print (end_time - start_time)
        print ('best fitness:')
        print (bestchrom[0])
        print ('best path:')
    best_path = fitness( bestchrom[1], cpoints, start_point, end_point, tmax )[1]
    if verbose:
        print ([ x[3] for x in best_path ])

        print ('their stuff:')
    stuff = oph.initialize( oph.ell_sub( tmax, start_point, end_point, cpoints )
    , start_point, end_point, tmax )[0]
    if verbose:
        print ('fitness:', sum( [ x[2] for x in stuff ] ))
        print ('my stuff:')
    stuff2 = oph.ellinit_replacement( cpoints, start_point, end_point, tmax )
    if verbose:
        print ('fitness:', sum( [ x[2] for x in stuff2 ] ))
        print ('checking correctness...')
    total_distance = ( oph.distance( start_point, cpoints[ best_path[ 1                    ][3] - 2 ] ) + 
                       oph.distance( end_point,   cpoints[ best_path[ len( best_path ) - 2 ][3] - 2 ] ) )
    for i in range( 1, len( best_path ) - 3 ):
        total_distance += oph.distance( cpoints[ best_path[ i     ][3] - 2 ], 
                                        cpoints[ best_path[ i + 1 ][3] - 2 ] )
    if verbose:
        print ('OK' if total_distance <= tmax else 'not OK')
        print ('tmax:          ', tmax)
        print ('total distance:', total_distance)
    if return_sol:
        return ( bestchrom[0], best_path, end_time - start_time )
    return ( bestchrom[0], end_time - start_time )

if( __name__ ==  '__main__' ):
    debug = True if 'd' in sys.argv else False
    run_alg( open( sys.argv[1] ), int( sys.argv[2] ), int( sys.argv[3] ) )
else:
    debug = False
