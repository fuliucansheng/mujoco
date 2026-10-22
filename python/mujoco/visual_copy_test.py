"""Regression for every exposed mjVisual headlight/RGBA field and its storage."""

import copy
import gc
import unittest

import numpy as np

import mujoco

RGBA_FIELDS = (
    'fog',
    'haze',
    'force',
    'inertia',
    'joint',
    'actuator',
    'actuatornegative',
    'actuatorpositive',
    'com',
    'camera',
    'light',
    'selectpoint',
    'connect',
    'contactpoint',
    'contactforce',
    'contactfriction',
    'contacttorque',
    'contactgap',
    'rangefinder',
    'constraint',
    'slidercrank',
    'crankbroken',
    'frustum',
    'bv',
    'bvactive',
)
HEADLIGHT_FIELDS = ('ambient', 'diffuse', 'specular')


class VisualCopyTest(unittest.TestCase):

  def check_visual_copy(self, owner_kind, substructure, copier):
    model = None
    if owner_kind == 'model':
      model = mujoco.MjModel.from_xml_string('<mujoco/>')
      visual = model.vis
    else:
      model = mujoco.MjModel.from_xml_string('<mujoco/>')
      visual = copy.copy(model.vis)
    original = getattr(visual, substructure)
    fields = HEADLIGHT_FIELDS if substructure == 'headlight' else RGBA_FIELDS
    width = 3 if substructure == 'headlight' else 4
    exposed = {
        name
        for name in dir(original)
        if not name.startswith('_')
        and isinstance(getattr(original, name), np.ndarray)
    }
    self.assertEqual(exposed, set(fields))
    expected = {}
    for index, name in enumerate(fields):
      view = getattr(original, name)
      self.assertEqual(view.shape, (width,), name)
      self.assertEqual(view.dtype, np.dtype('float32'), name)
      values = np.arange(width, dtype=np.float32) + np.float32(index + 1)
      view[:] = values
      expected[name] = values.copy()
    if substructure == 'headlight':
      original.active = 1
    cloned = copier(original)
    self.assertIsNot(cloned, original)
    for name in fields:
      source_view = getattr(original, name)
      cloned_view = getattr(cloned, name)
      self.assertEqual(cloned_view.shape, (width,), name)
      np.testing.assert_array_equal(cloned_view, expected[name])
      self.assertFalse(np.shares_memory(source_view, cloned_view), name)
      cloned_view[:] += 100
      np.testing.assert_array_equal(source_view, expected[name])
    if substructure == 'headlight':
      self.assertEqual(cloned.active, 1)
      cloned.active = 0
      self.assertEqual(original.active, 1)
    # Assignment updates existing views; it does not alias the copied owner.
    live_views = {name: getattr(original, name) for name in fields}
    setattr(visual, substructure, cloned)
    for name in fields:
      np.testing.assert_array_equal(live_views[name], expected[name] + 100)
      self.assertTrue(
          np.shares_memory(live_views[name], getattr(original, name))
      )
      self.assertFalse(
          np.shares_memory(live_views[name], getattr(cloned, name))
      )
      live_views[name][:] += 10
      np.testing.assert_array_equal(
          getattr(original, name), expected[name] + 110
      )
      np.testing.assert_array_equal(getattr(cloned, name), expected[name] + 100)
    if substructure == 'headlight':
      self.assertEqual(original.active, 0)
    # Array views retain storage after their model/wrapper owners are dropped.
    cloned_views = {name: getattr(cloned, name) for name in fields}
    del cloned, original, visual, model
    gc.collect()
    for name in fields:
      np.testing.assert_array_equal(live_views[name], expected[name] + 110)
      np.testing.assert_array_equal(cloned_views[name], expected[name] + 100)
      live_views[name][0] = -7
      self.assertNotEqual(cloned_views[name][0], -7)


def make_test(owner_kind, substructure, copier):
  def test(self):
    self.check_visual_copy(owner_kind, substructure, copier)

  return test


for _owner in ('owned_copy', 'model'):
  for _substructure in ('headlight', 'rgba'):
    for _copy_name, _copier in (
        ('copy', copy.copy),
        ('deepcopy', copy.deepcopy),
    ):
      setattr(
          VisualCopyTest,
          f'test_{_owner}_{_substructure}_{_copy_name}',
          make_test(_owner, _substructure, _copier),
      )

if __name__ == '__main__':
  unittest.main(verbosity=2)
