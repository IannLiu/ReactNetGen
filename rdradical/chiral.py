from __future__ import print_function
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

from rdradical.utils import vprint, parity4, PLEVEL

def nb_H_Num(atom):
    nb_Hs = 0
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == 'H':
            nb_Hs = nb_Hs + 1
    return(nb_Hs)

def template_atom_could_have_been_tetra(a, strip_if_spec=False, cache=True):
    '''Could this atom have been a tetrahedral center?
    If yes, template atom is considered achiral and will not match a chiral rct
    If no, the tempalte atom is auxilliary and we should not use it to remove
    a matched reaction. For example, a fully-generalized terminal [C:1] 
    
    Note: If yes, the tetra has not been specified, therefore, we should not specified
    this atom again, and thus it can be used to remove a matched reaction. However, 
    if this atom is terminal, we should not use this atom to do so.

    Args:
        a (rdkit.Chem.rdchem.Atom): RDKit atom
        strip_if_spec (bool, optional): Defaults to False.
        cache (bool, optional): Defaults to True.

    Returns:
        bool: Returns True if this atom have been a tetrahedral center
    '''
    H_number = nb_H_Num(a)
    if a.HasProp('tetra_possible'):
        return a.GetBoolProp('tetra_possible')
    if a.GetDegree()-H_number < 3 or a.GetDegree() == 3:
        if cache:
            a.SetBoolProp('tetra_possible', False)
        if strip_if_spec: # Clear chiral tag in case improperly set
            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
        return False 
    if cache:
        a.SetBoolProp('tetra_possible', True)
    return True 



def copy_chirality(a_src, a_new):
    '''Copy chirality from a_src to a_new

    Args:
        a_src (rdkit.Chem.rdchem.Atom): Source RDKit atom
        a_new (rdkit.Chem.rdchem.Atom): RDKit atom to have chirality changed

    Returns:
        None

    '''
    # Not possible to be a tetrahedral center anymore?
    if a_new.GetDegree() - nb_H_Num(a_new) < 3:
        return
    if a_new.GetDegree() - nb_H_Num(a_new) == 3 and \
            any(b.GetBondType() != BondType.SINGLE for b in a_new.GetBonds()):
        return
    if PLEVEL >= 3: print('For mapnum {}, copying src {} chirality tag to new'.format(
        a_src.GetAtomMapNum(), a_src.GetChiralTag()))
    a_new.SetChiralTag(a_src.GetChiralTag())
    
    if atom_chirality_matches(a_src, a_new) == -1:
        if PLEVEL >= 3: print('For mapnum {}, inverting chirality'.format(a_new.GetAtomMapNum()))
        a_new.InvertChirality()
        
def atom_chirality_matches(a_tmp, a_mol):
    '''
    Checks for consistency in chirality between a template atom and a molecule atom.

    Also checks to see if chirality needs to be inverted in copy_chirality

    Args:
        a_tmp (rdkit.Chem.rdchem.Atom): RDKit Atom
        a_mol (rdkit.Chem.rdchem.Mol): RDKit Mol

    Returns:
        int: Integer value of match result
            +1 if it is a match and there is no need for inversion (or ambiguous)
            -1 if it is a match but they are the opposite
            0 if an explicit NOT match
            2 if ambiguous or achiral-achiral
    '''
    if a_mol.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
        if a_tmp.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
            if PLEVEL >= 3: print('atom {} is achiral & achiral -> match'.format(a_mol.GetAtomMapNum()))
            return 2 # achiral template, achiral molecule -> match
        # What if the template was chiral, but the reactant isn't just due to symmetry?
        if not a_mol.HasProp('_ChiralityPossible'):
            # It's okay to make a match, as long as the product is achiral (even
            # though the product template will try to impose chirality)
            if PLEVEL >= 3: print('atom {} is specified in template, but cant possibly be chiral in mol'.format(a_mol.GetAtomMapNum()))
            return 2 # This means that we should not use it to remove the generated products.

        # Discussion: figure out if we want this behavior - should a chiral template
        # be applied to an achiral molecule? For the retro case, if we have
        # a retro reaction that requires a specific stereochem, return False;
        # however, there will be many cases where the reaction would probably work
        if PLEVEL >= 3: print('atom {} is achiral in mol, but specified in template'.format(a_mol.GetAtomMapNum()))
        return 0
    if a_tmp.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
        if PLEVEL >= 3: print('Reactant {} atom chiral, rtemplate achiral...'.format(a_tmp.GetAtomMapNum()))
        if template_atom_could_have_been_tetra(a_tmp):
            if PLEVEL >= 3: print('...and that atom could have had its chirality specified! no_match')
            return 0
        if PLEVEL >= 3: print('...but the rtemplate atom could not have had chirality specified, match anyway')
        return 2
    # both two aotms are specificed @@ or @
    mapnums_tmp = [a.GetAtomMapNum() for a in a_tmp.GetNeighbors()]    
    mapnums_mol = [a.GetAtomMapNum() for a in a_mol.GetNeighbors()]

    # When there are fewer than 3 heavy neighbors, chirality is ambiguous...
    if len(mapnums_tmp)- nb_H_Num(a_tmp)< 3 or len(mapnums_mol) - nb_H_Num(a_mol) < 3:
        return 2

    # Degree of 3 -> remaining atom is a hydrogen, add to list
    # However, due to explict Hs with map number, add -1 is neglected.
    """
    if len(mapnums_tmp) < 4:
        mapnums_tmp.append(-1) # H
    if len(mapnums_mol) < 4:
        mapnums_mol.append(-1) # H
    """
    # Now both two aotms are specificed @@ or @. We need to compare the source atom
    # and new atom to decide whether the chiral tag should be inverted.
    # !! Obtain the party !! In this case, we order the neighbors' map number from small to large
    # Then, couting the steps we need. 
    try:
        if PLEVEL >= 10: print(str(mapnums_tmp))
        if PLEVEL >= 10: print(str(mapnums_mol))
        if PLEVEL >= 10: print(str(a_tmp.GetChiralTag()))
        if PLEVEL >= 10: print(str(a_mol.GetChiralTag()))
        only_in_src = [i for i in mapnums_tmp if i not in mapnums_mol][::-1] # reverse for popping
        only_in_mol = [i for i in mapnums_mol if i not in mapnums_tmp]
        if len(only_in_src) <= 1 and len(only_in_mol) <= 1:
            tmp_parity = parity4(mapnums_tmp)
            mol_parity = parity4([i if i in mapnums_tmp else only_in_src.pop() for i in mapnums_mol])
            if PLEVEL >= 10: print(str(tmp_parity))
            if PLEVEL >= 10: print(str(mol_parity))
            parity_matches = tmp_parity == mol_parity
            tag_matches = a_tmp.GetChiralTag() == a_mol.GetChiralTag()
            chirality_matches = parity_matches == tag_matches
            if PLEVEL >= 2: print('mapnum {} chiral match? {}'.format(a_tmp.GetAtomMapNum(), chirality_matches))
            return 1 if chirality_matches else -1
        else:
            if PLEVEL >= 2: print('mapnum {} chiral match? Based on mapnum lists, ambiguous -> True'.format(a_tmp.GetAtomMapNum()))
            return 2 # ambiguous case, just return for now

    except IndexError as e:
        print(a_tmp.GetPropsAsDict())
        print(a_mol.GetPropsAsDict())
        print(a_tmp.GetChiralTag())
        print(a_mol.GetChiralTag())
        print(str(e))
        print(str(mapnums_tmp))
        print(str(mapnums_mol))
        raise KeyError('Pop from empty set - this should not happen!')