"""Tools to read/print basis sets into common formats"""

import re
import numpy as np


class Primitive:
    def __init__(self, exponent, coeffs, stype):
        """Initialize Primitive object.

        Parameters
        ----------
        exponent : str
            Exponent of a primitive.
        coeffs : list(str)
            All the coefficients of the primitives with the same exponent
        stype : str
            Type of shell/orbital `S`, `P`, `SP`, etc.
        """
        self.exponent = exponent
        self.coeffs = coeffs
        self.stype = stype


class BShell:
    def __init__(self, primitives, stype):
        """Initialize BShell object.

        Parameters
        ----------
        primitives : str
            List of `Primitive` objects for each basis shell.
        stype : str
            Type of shell/orbital `S`, `P`, `SP`, etc.
        """
        if not isinstance(primitives, (list, Primitive)):
            raise TypeError('The primitives must be instance of `Primitive class`')
        self.primitives = primitives
        self.stype = stype


def find_uniques(linp):
    new_list = []
    for element in linp:
        if element not in new_list:
            new_list.append(element)
    return new_list


def parse_bas_basis(basis):
    """Parse a basis in Gaussian/Q-Chem format.

    Parameters
    ----------
    basis : str
        String with information of the basis set.
    """
    new_shell = re.compile(r'\w{,2}\s+1.00$')
    new_element = re.compile(r'([a-z]{,2})\s+0$')
    endre = re.compile(r'\*\*\*\*')

    lines = basis.splitlines()
    epositions = []
    spositions = []
    ends = []
    comment = []
    for n, line in enumerate(lines):
        element = new_element.search(line, re.IGNORECASE)
        end = endre.search(line)
        shell = new_shell.search(line, re.IGNORECASE)
        if element:
            epositions.append(n)
        elif shell:
            spositions.append(n)
        elif end:
            ends.append(n)
        else:
            if line.startswith('!'):
                comment.append(line[1:])
    comment = ''.join(comment)
    print(epositions)
    print(ends)
    if len(epositions) != len(ends):
        raise ValueError('Wrong definition of at least one element.')
    elements = []
    exponents = []
    coefficients = []
    stypes = []
    # First save all the numbers
    for i, es in enumerate(epositions):
        beg = es + 1
        end = ends[i]
        shells = [s for s in spositions if beg <= s < end]
        elements.append(lines[es].split()[0])
        exponents.append([])
        coefficients.append([])
        stypes.append([])
        for j in shells:
            stype = lines[j].split()[0]
            nprims = lines[j].split()[1]
            for prim in range(int(nprims)):
                j +=1
                data = lines[j].split()
                exp = data[0]
                coeffs = data[1:]
                exponents[-1].append(exp)
                coefficients[-1].append(coeffs)
                stypes[-1].append(stype)
    # Place together the same functions
    bshells = []
    for i, element in enumerate(elements):
        bshells.append([])
        sprims_rep = []
        pprims_rep = []
        dprims_rep = []
        fprims_rep = []
        gprims_rep = []
        hprims_rep = []
        sprims = []
        pprims = []
        dprims = []
        fprims = []
        gprims = []
        hprims = []
        # Find repeated terms
        uniques = find_uniques(exponents[i])
        for j, exp in enumerate(uniques):
            # find index of repeated atom
            fin_coeffs = []
            rtimes = exponents[i].count(exp)
            beg = 0
            ind = exponents[i].index(exp, beg)
            if rtimes > 1 or abs(float(coefficients[i][ind][0])-1.0000) > 1e-8:
                for t in range(rtimes):
                    ind = exponents[i].index(exp, beg)
                    fin_coeffs += coefficients[i][ind]
                    stype = stypes[i][ind]
                    beg = ind+1
                if 'S' in stype:
                    sprims_rep.append(Primitive(exp, fin_coeffs, stype))
                if 'P' in stype:
                    pprims_rep.append(Primitive(exp, fin_coeffs, stype))
                if 'D' in stype:
                    dprims_rep.append(Primitive(exp, fin_coeffs, stype))
                if 'F' in stype:
                    fprims_rep.append(Primitive(exp, fin_coeffs, stype))
                if 'G' in stype:
                    gprims_rep.append(Primitive(exp, fin_coeffs, stype))
                if 'H' in stype:
                    hprims_rep.append(Primitive(exp, fin_coeffs, stype))
            else:
                ind = exponents[i].index(exp, beg)
                stype = stypes[i][ind]
                if 'S' in stype:
                    sprims.append(Primitive(exp, coefficients[i][ind], stype))
                if 'P' in stype:
                    pprims.append(Primitive(exp, coefficients[i][ind], stype))
                if 'D' in stype:
                    dprims.append(Primitive(exp, coefficients[i][ind], stype))
                if 'F' in stype:
                    fprims.append(Primitive(exp, coefficients[i][ind], stype))
                if 'G' in stype:
                    gprims.append(Primitive(exp, coefficients[i][ind], stype))
                if 'H' in stype:
                    hprims.append(Primitive(exp, coefficients[i][ind], stype))
        # Save shells by stype
        if len(sprims_rep) >= 1:
            bshells[-1].append(BShell(sprims_rep, 'S'))
        if len(sprims) >= 1:
            for p in sprims:
                bshells[-1].append(BShell([p], 'S'))
        if len(pprims_rep) >= 1:
            bshells[-1].append(BShell(pprims_rep, 'P'))
        if len(pprims) >= 1:
            for p in pprims:
                bshells[-1].append(BShell([p], 'P'))
        if len(dprims_rep) >= 1:
            bshells[-1].append(BShell(dprims_rep, 'D'))
        if len(dprims) >= 1:
            for p in dprims:
                bshells[-1].append(BShell([p], 'D'))
        if len(fprims_rep) >= 1:
            bshells[-1].append(BShell(fprims_rep, 'F'))
        if len(fprims) >= 1:
            for p in fprims:
                bshells[-1].append(BShell([p], 'F'))
        if len(gprims_rep) >= 1:
            bshells[-1].append(BShell(gprims_rep, 'G'))
        if len(gprims) >= 1:
            for p in gprims:
                bshells[-1].append(BShell([p], 'G'))
        if len(hprims_rep) >= 1:
            bshells[-1].append(BShell(hprims_rep, 'H'))
        if len(hprims) >= 1:
            for p in hprims:
                bshells[-1].append(BShell([p], 'H'))
    return comment, elements, bshells


def write_bas_basis(elements, bshells, bname, comment=None):
    """Write to file the basis set information in Gaussian format.

    Parameters
    ----------
    elements : list(str)
        The elements for which the basis set is for.
    bshells : list(list(BShell))
        Information of basis shell, that contains each primitive of each element.
    bname : str
        Name of basis set without extension.
    comment : str
        Any extra comment to be added to the top of the file
    """
    if len(elements) != len(bshells):
        raise ValueError('Number of elements and group of primitives do not match.')
    bfile = bname+'.bas'
    with open(bfile, 'w') as bf:
        if comment:
            bf.write(comment)
        bf.write("! Basis set %s created by FDETaco\n" % bname)
        bf.write('\n\n')
        for ie, element in enumerate(elements):
            bf.write(' {}    0\n'.format(element))
            shells = bshells[ie]
            for shell in shells:
                nprims = len(shell.primitives)
                stype = shell.stype
                bf.write(' %s    %d   1.00\n' % (stype, nprims))
                bstring = ''
                for prim in shell.primitives:
                    bstring += '{:>16}'.format(prim.exponent)
                    for coeff in prim.coeffs:
                        bstring += '{:>20}'.format(coeff)
                    bstring += '\n'
                bf.write(bstring)
            bf.write(' ****\n')


# def parse_nwchem_basis(basis):
#     """Parse a basis in Gaussian/Q-Chem format.
# 
#     Parameters
#     ----------
#     basis : str
#         String with information of the basis set.
#     """
#     new_shell = re.compile(r'^\s+\w{,2}\s+[S, P, D, F, G, H, I]')
#     new_element = re.compile(r'\#BASIS SET:*\-\>*')
#     # Split in lines
#     lines = basis.splitlines()
#     epositions = []
#     spositions = []
#     ends = []
#     comment = []
#     for n, line in enumerate(lines):


def write_nwchem_basis(elements, bshells, bname, comment=None):
    """Write to file the basis set information in NWChem format.

    Parameters
    ----------
    elements : list(str)
        The elements for which the basis set is for.
    bshells : list(list(BShell))
        Information of basis shell, that contains each primitive of each element.
    bname : str
        Name of basis set without extension.
    comment : str
        Any extra comment to be added to the top of the file
    """
    if len(elements) != len(bshells):
        raise ValueError('Number of elements and group of primitives do not match.')
    bfile = bname+'.nwchem'
    with open(bfile, 'w') as bf:
        if comment:
            bf.write('# '+comment+'\n')
        bf.write("# Basis set %s created by FDETaco\n" % bname)
        bf.write('\n\n')
        bf.write("BASIS \"ao basis\" PRINT\n")
        for ie, element in enumerate(elements):
            shells = bshells[ie]
            bstring = ''
            tots = 0; totp = 0; totd = 0
            totf = 0; totg = 0; toth = 0
            fins = 0; finp = 0; find = 0
            finf = 0; fing = 0; finh = 0
            for shell in shells:
                stype = shell.stype
                if 'S' in stype:
                    tots += len(shell.primitives)
                    fins += len(shell.primitives[0].coeffs)
                if 'P' in stype:
                    totp += len(shell.primitives)
                    finp += len(shell.primitives[0].coeffs)
                if 'D' in stype:
                    totd += len(shell.primitives)
                    find += len(shell.primitives[0].coeffs)
                if 'F' in stype:
                    totf += len(shell.primitives)
                    finf += len(shell.primitives[0].coeffs)
                if 'G' in stype:
                    totg += len(shell.primitives)
                    fing += len(shell.primitives[0].coeffs)
                if 'H' in stype:
                    toth += len(shell.primitives)
                    finh += len(shell.primitives[0].coeffs)
                bstring += '{:6} {}'.format(element, stype)
                bstring += '\n'
                for prim in shell.primitives:
                    bstring += '{:>16}'.format(prim.exponent)
                    for coeff in prim.coeffs:
                        bstring += '{:>20}'.format(coeff)
                    bstring += '\n'

            # Make comment string
            totstring = ''
            if tots > 0:
                totstring += '%ss' % tots
            if totp > 0:
                totstring +=',%sp' % totp
            if totd > 0:
                totstring +=',%sd' % totd
            if totf > 0:
                totstring +=',%sf' % totf
            if totg > 0:
                totstring +=',%sg' % totg
            if toth > 0:
                totstring +=',%sh' % toth
            finstring = ''
            if fins > 0:
                finstring += '%ss' % fins
            if finp > 0:
                finstring +=',%sp' % finp
            if find > 0:
                finstring +=',%sd' % find
            if finf > 0:
                finstring +=',%sf' % finf
            if fing > 0:
                finstring +=',%sg' % fing
            if finh > 0:
                finstring +=',%sh' % finh
            bf.write("#BASIS SET: (%s) -> [%s]\n" % (totstring, finstring))
            # Finally print basis
            bf.write(bstring)
        bf.write('END\n')
        bf.write('\n')
