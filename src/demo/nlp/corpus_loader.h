/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_NLP_CORPUS_LOADER_H_
#define DEMO_NLP_CORPUS_LOADER_H_

#include "../../utils/laminar_utils.h"

// 64 chars in consideration, using one-hot encoding
static const constexpr int CORPUS_ONE_HOT_DIM = 64;

/**
 * Load int class labels from a preprocessed corpus
 * 64 classes of chars ('A' - 'Z', 'a' - 'z', and 12 special chars)
 */
class CorpusLoader
{
public:
	CorpusLoader(std::string filePath) :
		ifs(filePath, std::ios::binary)
	{
		LMN_ASSERT_THROW(ifs.is_open(),
			DataException(filePath + " corpus file not found"));

		// the first 4 bytes (int) is total number of chars in this file
		ifs.read((char *) &corpusSize, sizeof(corpusSize));
	}

	int size() const
	{
		return this->corpusSize;
	}

	/**
	 *
	 * @param n number of chars to be loaded
	 * @return
	 */
	std::vector<int> load(int n)
	{
		LMN_ASSERT_THROW(!ifs.eof(),
			DataException("CorpusLoader::load() fails: EOF"));

		std::vector<int> ans;
		for (int i = 0; i < n; ++i)
		{
			unsigned char c = 0;
			ifs.read((char *) &c, sizeof(c));
			ans.push_back(c);
		}
		return ans;
	}

	/**
	 * Break corpus stream into segments of certain length
	 * @param numOfSegments
	 * @param segmentLength
	 * @return
	 */
	std::vector<std::vector<int>>
			load_segment(int numOfSegments, int segmentLength)
	{
		std::vector<std::vector<int>> ans;

		for (int b = 0; b < numOfSegments; ++b)
			ans.push_back(this->load(segmentLength));

		return ans;
	}

	void close()
	{
		ifs.close();
	}

	/**
	 * Laminar defined training corpus mapping
	 */
	static char code2char(int code)
	{
		// first 26 are 'A' - 'Z'
		if (0 <= code && code <= 25)
			return char(code + 65);
		// next 26 are 'a' - 'z'
		else if (26 <= code && code <= 51)
			return char(code + 71);
		else
			switch (code)
			{
			case 52 : return ' ';
			case 53 : return '!';
			case 54 : return '"';
			case 55 : return '\'';
			case 56 : return '(';
			case 57 : return ')';
			case 58 : return ',';
			case 59 : return '-';
			case 60 : return '.';
			case 61 : return ':';
			case 62 : return ';';
			case 63 : return '?';

			default :
				throw DataException(
				"CorpuseLoader conversion error: unknown code " + std::to_string(code));
			}
	}

	/**
	 * Reverse of code2char
	 */
	static int char2code(char ch)
	{
		// first 26 are 'A' - 'Z'
		if ('A' <= ch && ch <= 'Z')
			return int(ch - 65);
		// next 26 are 'a' - 'z'
		else if ('a' <= ch && ch <= 'z')
			return int(ch - 71);
		else
			switch (ch)
			{
			case ' ' : return 52;
			case '!' : return 53;
			case '"' : return 54;
			case '\'': return 55;
			case '(' : return 56;
			case ')' : return 57;
			case ',' : return 58;
			case '-' : return 59;
			case '.' : return 60;
			case ':' : return 61;
			case ';' : return 62;
			case '?' : return 63;

			default :
				throw DataException(
				"CorpuseLoader conversion error: unknown char " + std::to_string(ch));
			}
	}


	/**
	 * Concat the chars to a string
	 */
	static std::string code2str(std::vector<int> codes)
	{
		std::string ans = "";
		for (int x : codes)
			ans += code2char(x);
		return ans;
	}

private:
	std::ifstream ifs;

	int corpusSize = 0;
};


#endif /* DEMO_NLP_CORPUS_LOADER_H_ */
