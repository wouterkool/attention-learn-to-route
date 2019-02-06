import math

def distance( p1, p2 ):
    return math.sqrt( ( p1[0] - p2[0] ) ** 2 + ( p1[1] - p2[1] ) ** 2 )

#returns a path (list of points) through s with high value
def ellinit_replacement( s1, start_point, end_point, tmax ):
    s = list( s1 )
    path = [ start_point, end_point ]
    length = distance( start_point, end_point )
    found = True
    while( found == True and len( s ) > 0 ):
        min_added_length = -1
        max_added_reward = 0
        for j in range( len( s ) ):
            for k in range( len( path ) - 1 ):
                added_length = ( distance( path[ k ], s[ j ] ) + 
                                 distance( path[ k + 1 ], s[ j ] ) - 
                                 distance( path[ k ], path[ k + 1 ] ) ) # optimize later
                if( length + added_length < tmax and s[ j ][2] > max_added_reward ):
                    min_added_length = added_length
                    max_added_reward = s[ j ][2]
                    minpoint = j
                    pathpoint = k + 1
        if( min_added_length > 0 ):
            #add to path
            path.insert( pathpoint, s.pop( minpoint ) )
            length = length + min_added_length
        else:
            found = False
    return path

#returns a list of L paths with the best path in the first position
#by weight rather than length
def init_replacement( s1, start_point, end_point, tmax ):
    s = list( s1 )
    L = len( s ) if len( s ) <= 10 else 10
    if( L == 0 ):
        #print 'something is probably wrong'
        #actually maybe not
        return [ [ start_point, end_point ] ]

    #decorate and sort by weight
    dsub = sorted( [ ( x[4], x ) for x in s ] )[::-1] #this is different
    ls = dsub[ :L ] 
    rest = dsub[ L: ]
    paths = []
    for i in range( L ):
        path = [ start_point, ls[ i ][1] , end_point ] 
        length = distance( path[0], path[1] ) + distance( path[1], path[2] )
        assert( length < tmax )
        arest = ls[ :i ] + ls[ i + 1: ] + rest
        arest = [ x[1] for x in arest ] #undecorate
        assert( len( arest ) + len( path ) == len( s ) + 2 )
        found = True
        while( found == True and len( arest ) > 0 ):
            min_added_length = -1
            max_weight = 0
            for j in range( len( arest ) ):
                for k in range( len( path ) - 1 ):
                    added_length = ( distance( path[ k ], arest[ j ] ) + 
                                     distance( path[ k + 1 ], arest[ j ] ) - 
                                     distance( path[ k ], path[ k + 1 ] ) ) # optimize later
                    if( length + added_length < tmax and arest[ j ][4] < max_weight ):
                        min_added_length = added_length
                        max_weight = arest[ j ][4]
                        minpoint = j
                        pathpoint = k + 1
            if( min_added_length > 0 ):
                #add to path
                path.insert( pathpoint, arest.pop( minpoint ) )
                length = length + min_added_length
            else:
                found = False
        if( length < tmax ):
            paths.append( path )

    assert( len( paths ) > 0 )
    return [ x[1] for x in sorted( [ ( sum( [ y[2] for y in z ] ), z ) for z in paths ] )[::-1] ]


#returns the subset of s that is on/in the ellipse defined by foci f1, f2 and the major axis
def ell_sub( axis, f1, f2, s ):
    result = []
    for item in s:
        if( distance( item, f1 ) + distance( item, f2 ) <= axis ):
            result.append( item )
    return result

#returns a list of L paths with the best path in the first position
def initialize( s, start_point, end_point, tmax ):
    L = len( s ) if len( s ) <= 10 else 10
    if( L == 0 ):
        return [ [ start_point, end_point ] ]

    dsub = sorted( [ ( distance( x, start_point ) + distance( x, end_point ), x ) for x in s ]
            )[::-1] #optimize later
    ls = dsub[ :L ] 
    rest = dsub[ L: ]
    paths = []
    for i in range( L ):
        path = [ start_point, ls[ i ][1] , end_point ] 
        length = ls[ i ][0]
        assert( length == distance( path[0], path[1] ) + distance( path[1], path[2] ) )
        arest = ls[ :i ] + ls[ i + 1: ] + rest
        arest = [ x[1] for x in arest ] #undecorate
        assert( len( arest ) + len( path ) == len( s ) + 2 )
        found = True
        while( found == True and len( arest ) > 0 ):
            min_added = -1
            for j in range( len( arest ) ):
                for k in range( len( path ) - 1 ):
                    added_length = ( distance( path[ k ], arest[ j ] ) + 
                                     distance( path[ k + 1 ], arest[ j ] ) - 
                                     distance( path[ k ], path[ k + 1 ] ) ) # optimize later
                    if( length + added_length < tmax and ( added_length < min_added or min_added < 0 ) ):
                        min_added = added_length
                        minpoint = j
                        pathpoint = k + 1
            if( min_added > 0 ):
                #add to path
                path.insert( pathpoint, arest.pop( minpoint ) )
                length = length + min_added
            else:
                found = False
        paths.append( path )

    assert( len( [ x[1] for x in sorted( [ ( sum( [ y[2] for y in z ] ), z ) for z in paths ]
        )[::-1] ] ) > 0 )
    return [ x[1] for x in sorted( [ ( sum( [ y[2] for y in z ] ), z ) for z in paths ] )[::-1] ]

