use bellpepper_core::{boolean::Boolean, num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{
  gadgets::{
    r1cs::{conditionally_select_alloc_relaxed_r1cs, AllocatedRelaxedR1CSInstance},
    utils::{alloc_const, alloc_num_equals, conditionally_select},
  },
  traits::Group,
};

// return the element matched index
// WARNING: there is no check for index out of bound. By default will return first one
// FIXME use api `try_result` https://doc.rust-lang.org/stable/std/iter/trait.Iterator.html#method.try_reduce to fine tune function logic once stable
// TODO optimize this part to have raw linear-combination on variables to achieve less constraints
pub fn get_from_vec_alloc_relaxed_r1cs<G: Group, CS: ConstraintSystem<<G as Group>::Base>>(
  mut cs: CS,
  a: &[AllocatedRelaxedR1CSInstance<G>],
  target_index: &AllocatedNum<G::Base>,
) -> Result<(AllocatedNum<G::Base>, AllocatedRelaxedR1CSInstance<G>), SynthesisError> {
  let mut a = a.iter().enumerate();
  let first = (
    alloc_const(cs.namespace(|| "i_const 0 allocated"), G::Base::from(0u64))?,
    a.next()
      .ok_or_else(|| SynthesisError::IncompatibleLengthVector("empty vec length".to_string()))?
      .1
      .clone(),
  );
  let selected = a.try_fold(first, |matched, (i, candidate)| {
    let i_const = alloc_const(
      cs.namespace(|| format!("i_const {:?} allocated", i)),
      G::Base::from(i as u64),
    )?;
    let equal_bit = Boolean::from(alloc_num_equals(
      cs.namespace(|| format!("check {:?} equal bit", i_const.get_value().unwrap())),
      &i_const,
      target_index,
    )?);
    let next_matched_index = conditionally_select(
      cs.namespace(|| {
        format!(
          "select on index namespace {:?}",
          i_const.get_value().unwrap()
        )
      }),
      &i_const,
      &matched.0,
      &equal_bit,
    )?;
    let next_matched_allocated = conditionally_select_alloc_relaxed_r1cs(
      cs.namespace(|| {
        format!(
          "select on index namespace {:?}",
          i_const.get_value().unwrap()
        )
      }),
      candidate,
      &matched.1,
      &equal_bit,
    )?;
    Ok::<(AllocatedNum<G::Base>, AllocatedRelaxedR1CSInstance<G>), SynthesisError>((
      next_matched_index,
      next_matched_allocated,
    ))
  })?;
  Ok(selected)
}
