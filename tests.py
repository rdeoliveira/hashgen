from hashgen import good_tag
import unittest

class TestGoodTag(unittest.TestCase):
	'''
	Test cases for ``good_tag``.
	'''

	def test_pronoun_us(self):
		'''
		Tests that the tag "us", when a pronoun, should be rejected.
		'''
		self.assertEqual(good_tag('us', 'PRP', []), False)

	def test_name_us(self):
		'''
		Tests that the tag "us", when a proper name (as in "the U.S."), should be accepted.
		'''
		self.assertEqual(good_tag('us', 'NNP', []), True)

	def test_letters_only(self):
		'''
		Tests that a tag with only letters should be accepted.
		'''
		self.assertEqual(good_tag('route', '', []), True)

	def test_numbers_only(self):
		'''
		Tests that a tag with only numbers should be rejected.
		'''
		self.assertEqual(good_tag('66', '', []), False)

	def test_aplha_numeric(self):
		'''
		Tests that a tag with mixed letters and numbers should be accepted.
		'''
		self.assertEqual(good_tag('route66', '', []), True)

	def test_is_stopword(self):
		'''
		Tests that a black-listed tag should be rejected.
		'''
		self.assertEqual(good_tag('route', '', ['route']), False)

	def test_not_stopword(self):
		'''
		Tests that a non-black-listed tag should be accepted.
		'''
		self.assertEqual(good_tag('route', '', []), True)

if __name__ == "__main__":
    unittest.main()