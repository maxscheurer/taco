"""
Test translate functions.
"""
import re
import pytest
import numpy as np

from taco.translate.order import get_sort_list
from taco.translate.order import transform
from taco.translate.tools import triangular2square, reorder_matrix
from taco.translate.tools import parse_matrix_molcas, parse_matrices
from taco.translate.tools import parse_orbitals_molcas
from taco.testdata.cache import cache


def test_get_sort_list():
    """Test sorting list function."""
    natoms = 3
    orders = [[0, 1, 2, 3], [0, 1, 2, 3, 6, 4, 7, 5, 8, 9],
              [0, 1, 2, 3, 6, 4, 7, 5, 8, 9]]
    ref_list = [0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13,
                14, 15, 16, 17, 20, 18, 21, 19, 22, 23]
    sort_list = get_sort_list(natoms, orders)
    assert ref_list == sort_list


def test_transform():
    """Test function to reorder matrices."""
    ref = np.arange(1, 26, 1).reshape(5, 5)
    ref = (ref + ref.T) - np.diag(ref.diagonal())
    # Original array
    # [[ 1  8 14 20 26]
    # [ 8  7 20 26 32]
    # [14 20 13 32 38]
    # [20 26 32 19 44]
    # [26 32 38 44 25]]
    ref_ordered = np.array([[1,  8,  14, 26, 20],
                            [8,  7,  20, 32, 26],
                            [14, 20, 13, 38, 32],
                            [26, 32, 38, 25, 44],
                            [20, 26, 32, 44, 19]])
    atoms = 2
    orders = [[0, 1], [0, 2, 1]]
    result = transform(ref, atoms, orders)
    assert np.allclose(ref_ordered, result)


def test_parse_matrix_molcas():
    """Reading matrices from molcas RUNASCII."""
    n0 = 'n'
    lines0 = 'line'
    with pytest.raises(TypeError):
        parse_matrix_molcas(lines0, n0)
    with pytest.raises(TypeError):
        parse_matrix_molcas([lines0], n0)
    lines = [' oasjnvriaksnc', '  3       2',
             '14.980    1230.0      984.0128']
    n = 0
    matrix = parse_matrix_molcas(lines, n)
    np.testing.assert_allclose(matrix, np.array([14.980, 1230.0, 984.0128]))


def test_parse_matrices_molcas():
    """Read matrices parser, OMolcas example."""
    fname = cache.files["molcas_runascii_co_h2o_cc-pvdz"]
    hook0 = 'hook'
    with pytest.raises(TypeError):
        parse_matrices(fname, hook0, 'molcas')
    hook = {'dipole': re.compile(r'\<(Dipole moment.)')}
    with pytest.raises(NotImplementedError):
        parse_matrices(fname, hook, 'qchem')
    parsed = parse_matrices(fname, hook, 'molcas')
    ref = np.array([0.165830742659920816, -0.228990590320634624e-01,
                    -0.196753869675605486e-01])
    np.testing.assert_allclose(ref, parsed['dipole'])


orbref = np.array([-1.09170286247068E-04, -3.66758942998072E-04,  2.16327337703413E-05,  1.26318309282575E-03,  3.26550829435096E-05,
                -4.37434762969739E-04,  4.43931690692416E-04, -2.76584913535365E-06, -4.03235296545293E-05,  5.98495343987971E-04,
                 4.27350010868954E-08, -1.58717574901880E-05,  3.18409314807086E-05,  3.74583868779968E-06,  3.95477462539479E-04,
                -7.49896381072481E-08,  1.67875863018379E-05,  2.78221111391544E-05, -1.82727557425233E-04, -3.25789070834698E-06,
                 2.98612196874529E-05, -3.21861324384450E-05,  1.93781310090749E-04, -6.88271168348026E-06, -5.39140711271363E-07,
                 1.74451122362151E-06, -1.37652264448550E-09,  1.88095373501380E-06,  5.14079551222001E-08, -5.15183405792441E-06,
                -6.84942194584801E-06, -1.65760873363140E-05,  1.99177126124212E-05, -1.20142209103604E-03,  4.75654127048409E-05,
                -2.06819554343108E-04, -3.45578835479653E-04, -2.58922838381469E-05,  7.01658370522225E-05,  4.00275731710636E-04,
                 1.02491599780629E-06, -6.03299823410294E-06, -6.22955023075480E-06, -8.44463043597033E-06,  1.78511438262681E-04,
                -4.84352914767101E-07,  8.07929059941296E-06,  1.83179913474041E-06,  2.24652547347664E-05, -1.97042327269976E-07,
                -1.87577198049925E-06,  2.89779125616542E-06, -9.69590145261100E-05,  1.82272048220305E-05,  1.26578754912494E-06,
                -6.23524434213025E-06, -3.34326667708259E-07,  8.86791941729378E-07,  2.81455823090981E-07,  4.26548701209726E-06,
                -1.22723534169353E-04, -4.02633286091807E-04,  4.47015157427014E-05,  8.62790716694433E-04,  2.88813276125836E-05,
                -1.38418931552858E-04, -4.94852559246139E-05, -3.78304819769211E-05,  2.95987914142914E-04, -3.14773642486752E-04,
                -4.03630299387013E-07,  1.75135779308273E-06, -6.98309803120675E-06,  2.04353261050456E-05,  9.62556500771604E-05,
                -5.95698179912133E-09,  1.16416154192377E-05,  1.90589438403690E-05, -1.14121008016992E-04, -5.77441707816697E-07,
                 2.11552924905560E-06,  1.20944423590158E-05, -2.23562193367573E-04, -1.97432699491016E-05, -1.02056143378063E-06,
                -8.23041194970277E-06, -9.16417236702375E-07, -3.10520187811381E-06,  6.43754619988964E-08,  1.34259461129044E-05,
                 2.65621420162972E-06,  5.77716913052789E-05, -1.16652561072865E-04, -1.02102256541963E-05,  3.31398187342893E-06,
                 8.65278775490550E-06, -1.60468534380698E-05, -2.56796647926376E-07, -1.68769275319335E-07,  6.06740309226792E-06,
                 8.38088449823525E-08,  7.25250888017628E-06, -5.36588815075548E-07, -4.41855459782086E-06, -4.12136712071965E-05,
                -1.54581984244807E-04,  3.83168898349323E-05, -2.25170477684133E-04, -9.84869705355529E-06,  2.50974974963503E-05,
                 5.95906304059300E-07, -2.49956325644862E-05,  1.02589981412863E-04,  1.82951738501263E-04, -8.82918722343653E-07,
                 3.06156371845832E-06,  3.61889507955654E-06, -3.11144322307781E-07, -1.25301929202677E-05,  7.49723991318117E-08,
                -1.56928616240566E-06, -1.90060735283732E-06,  1.84738105375867E-05,  1.03077090029732E-07, -6.72863732135452E-07,
                -1.42231160260200E-06,  2.42718109413361E-05,  5.03796605596497E-06,  1.30952888439607E-07,  3.67636357150704E-06,
                 7.44937188439151E-08, -1.33979336189871E-06,  1.92026283058588E-07, -3.54316656770778E-06, -1.93232243557910E-04,
                -6.00514490333890E-04,  1.25242127156335E-04,  5.69661563054391E-04,  3.20364603294853E-05, -1.06437891866008E-04,
                -5.86864381695904E-04, -1.09667904339018E-04,  4.23788711699860E-04,  2.83479643741779E-04, -1.29889733389773E-06,
                 3.79379068811265E-06, -2.23965187272326E-05,  1.37427528357434E-05, -6.74245451550165E-06, -5.65676661277266E-07,
                -1.71383633065639E-06,  3.37709950307949E-05, -6.41575193102489E-05, -2.38623939309231E-06,  1.42801158520109E-05,
                 3.64168122870448E-06,  1.89259221602935E-04, -1.48431618538738E-05, -1.10010771218299E-06,  8.29676374993628E-06,
                -1.17480617690714E-06, -1.36589823648397E-05,  9.97888163951185E-07,  1.28518093993514E-06, -6.05167188500508E-04,
                -2.27999516655291E-03,  3.97109496972274E-04,  7.92186830936370E-04, -2.76248155768323E-04, -2.41441198448278E-03,
                 1.44571350729874E-03,  1.83943672353961E-04,  1.55138993322694E-03, -9.99888935823414E-04, -1.00394866379360E-05,
                -8.11656299527725E-05,  4.65600002673429E-05,  3.73911221365821E-04,  1.47100982365049E-03,  1.31051715186803E-05,
                 6.43457390998021E-05,  2.25853503252653E-04,  4.50092888698469E-04, -1.90702562145616E-05, -3.79869245014632E-05,
                -1.64993674653983E-04, -6.59926888011544E-04,  8.06312401644102E-05,  8.08203121498521E-06, -1.25678338877162E-04,
                 1.54210114522538E-05,  1.87626682045813E-04, -1.50453319676625E-05,  6.77799448270819E-06, -9.47977241305669E-05,
                 3.73276660958564E-05,  4.90773466971342E-05, -1.66686218371690E-06,  5.08394228456025E-05,  1.84002155061806E-05,
                 4.71225217700299E-05,  3.44941795289656E-07,  3.27704363273706E-06, -9.78752305559867E-07, -1.02230067989173E-06,
                 1.53574767377963E-05, -2.09112760278866E-07,  2.25475247944375E-05, -1.74963091677287E-04, -5.43818941897523E-04,
                 1.16669094900336E-04,  4.64997424006756E-04,  1.22483457034669E-04, -4.54867183292772E-04, -6.90299979154890E-04,
                 1.27413622969417E-05, -5.27084878143096E-05,  3.33402400094431E-04,  5.16566160936521E-06, -2.09851529608038E-05,
                -2.55311389800322E-05,  9.54502154404291E-06, -1.77893907555715E-04, -1.30499414828753E-07, -5.84355511208698E-06,
                 3.22670789973102E-05, -5.58325582335563E-05, -3.03524514058298E-06, -2.01844700750888E-06, -1.66218081671386E-05,
                -6.20030639217324E-05, -1.36538315587708E-05, -8.28203293527484E-07,  9.24950147532009E-06, -9.00655123979388E-07,
                -1.21029280547545E-05,  6.91747598116006E-07, -4.68313567074605E-06, -6.73503567211367E-05,  5.93886000024584E-05,
                 1.91081826137430E-05, -2.15861848804995E-05, -9.99097979926379E-06, -3.20882187181552E-06, -4.00537459999801E-05,
                -1.39683455936708E-06, -1.08492528845131E-06, -1.80472484835634E-05, -1.20331072509635E-06,  1.28747871531055E-05,
                -2.08235959592370E-06, -1.49038261610570E-05, -8.05622667640751E-05, -5.75404758068400E-04,  2.56398353013361E-04,
                -4.90795703741649E-06, -2.30482748595714E-04, -6.00144311516632E-06, -1.56673973462290E-04, -2.89537349899832E-07,
                -1.22566594581353E-05, -1.55722169894612E-05, -8.27367193586325E-07,  5.60822847486677E-06, -7.67714731322180E-07,
                 9.59387022243271E-07, -9.98480145488439E-01,  3.63806738188227E-03, -1.59311502200707E-03, -1.09733686733348E-03,
                 4.38153776468335E-03, -4.20891506073203E-03, -1.65152685509816E-03, -2.84097288874127E-03,  2.73375313702930E-03,
                 1.07635646924843E-03,  1.50952906375568E-04, -1.45161314948364E-04, -5.63458768375176E-05,  8.84649523899592E-05,
                -6.62407149288273E-04,  1.64357049074848E-06, -2.47320862484283E-05,  1.02088268290550E-04, -3.53948121924012E-04,
                -8.27245420137795E-06,  2.99564170675220E-05, -4.14286466292350E-05,  2.91188482272188E-04,  8.14503447587788E-05,
                 5.48279767670057E-06, -4.05599914466046E-05,  5.15926544325961E-06,  6.16748116054191E-05, -3.05564559209809E-06,
                 1.41165331355157E-05])


def test_parse_orbitals_molcas():
    """Read matrices parser, OMolcas orbitals example."""
    fname = cache.files["molcas_uracilw1_natorbitals_cc-pvtz"]
    hook = {'orbital1': re.compile(r'\* ORBITAL    1    1')}
    parsed = parse_matrices(fname, hook, 'molcas', file_type='orbitals')
    np.testing.assert_allclose(orbref, parsed['orbital1'])


def test_molcas_parse_orbitals():
    fname = cache.files["molcas_uracilw1_natorbitals_cc-pvtz"]
    orbs = parse_orbitals_molcas(fname)
    np.testing.assert_allclose(orbref, orbs[0, :])


def print_density_matrix():
    """Read matrices parser, OMolcas orbitals example."""
    fname = cache.files["molcas_uracilw1_natorbitals_cc-pvtz"]
    hook = {'occupations' : re.compile(r'\* OCCUPATION NUMBERS')}
    parsed = parse_matrices(fname, hook, 'molcas', file_type='orbitals')
    occs = parsed['occupations']
    orbs = parse_orbitals_molcas(fname)
    dm = np.dot(occs*orbs.T, orbs)
    print(dm[8:, 8:])
    print(np.trace(dm[8:, 8:]))
#   print(np.sum(occs))
#   np.savetxt('dm_from_nos.txt', dm, delimiter='\n')
#   fname_mos = cache.files["molcas_uracilw1_morbitals_cc-pvtz"]
#   mos = parse_orbitals_molcas(fname_mos)
#   from taco.embedding.density import transform_density_matrix
#   dm_mos = transform_density_matrix(dm, mos)
#   print(dm_mos)
#   print(np.trace(dm_mos[8:, 8:]))


def test_triangular2square():
    """Test function to make full square matrix."""
    mat0 = [0, 1, 2, 3]
    n0 = 'number'
    mat1 = np.arange(0, 9)
    n1 = 4 
    with pytest.raises(TypeError):
        triangular2square(mat0, n1)
    with pytest.raises(TypeError):
        triangular2square(mat1, n0)
    with pytest.raises(ValueError):
        triangular2square(mat1, n1)
    mat = np.arange(0, 10)
    square = triangular2square(mat, n1)
    ref = np.array([[0, 0, 0, 0],
                    [1, 2, 0, 0],
                    [3, 4, 5, 0],
                    [6, 7, 8, 9]])
    np.testing.assert_allclose(square, ref)



def test_reoder_matrix():
    """Test re-ordering function for matrices."""
    inmat0 = [0, 1, 2]
    inprog0 = 3
    basis0 = 0
    atoms0 = np.array([0.2, 1.5])
    inmat1 = np.array([0, 1, 2])
    inprog1 = 'pyscf'
    basis1 = 'cc-pvdz'
    basis2 = 'cc-pvtz'
    atoms1 = np.array([2])
    with pytest.raises(TypeError):
        reorder_matrix(inmat0, inprog1, inprog1, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog0, inprog1, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog0, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog1, basis0, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog1, basis1, atoms0)
    outprog0 = 'gaussian'
    outprog1 = 'molcas'
    with pytest.raises(KeyError):
        reorder_matrix(inmat1, inprog1, outprog0, basis1, atoms1)
    with pytest.raises(KeyError):
        reorder_matrix(inmat1, inprog1, outprog1, basis2, atoms1)
    ref = np.array([[0, 1, 3, 6, 10],
                    [1, 2, 4, 7, 11],
                    [3, 4, 5, 8, 12],
                    [6, 7, 8, 9, 13],
                    [10, 11, 12, 13, 14]])
    finmat = reorder_matrix(ref, inprog1, outprog1, basis1, atoms1)
    np.testing.assert_allclose(finmat, ref)
    

if __name__ == "__main__":
#   test_get_sort_list()
#   test_transform()
#   test_parse_matrix_molcas()
#   test_parse_matrices_molcas()
#   test_triangular2square()
#   test_reoder_matrix()
#   test_parse_orbitals_molcas()
#   test_molcas_parse_orbitals()
    print_density_matrix()
