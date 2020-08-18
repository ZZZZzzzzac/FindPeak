/*!
 * \file peak_finder.cpp
 * \date 2019/01/21
 *
 * \author red
 * Contact: red.li@sigtrum.com
 *
 * \brief 
 *
 * TODO: long description
 *
 * \note
*/

#include ".\peak_finder.h"

std::vector<PeakInfo> findBandInRange(
	const ArrayMap<double> &x, double peakThreshold, int si, int se, int minBandPts=3)
{
	auto s = x.slice(si, se);
	auto mxi = (int)s.argmax();

	//
	int i = mxi - 1;
	while (i >= 0 && s[i] >= s[mxi] - peakThreshold)
		i -= 1;

	//
	int j = mxi + 1;
	while (j < (int)s.size() && s[j] >= s[mxi] - peakThreshold)
		j += 1;

	if (i < 0 && j == s.size())
		return std::vector<PeakInfo>(); //no peak found

	std::vector<PeakInfo> ipeaks;
	if (i >= minBandPts)
		ipeaks = findBandInRange(x, peakThreshold, si, si + i, minBandPts);

	std::vector<PeakInfo> peaks;
	if (i >= 0 && j < (int)s.size())
		peaks.push_back(std::make_pair(si + i, si + j));

	std::vector<PeakInfo> jpeaks;
	if (j < int(s.size() - minBandPts))
		jpeaks = findBandInRange(x, peakThreshold, si + j, se, minBandPts);

	ipeaks.insert(ipeaks.end(), peaks.begin(), peaks.end());
	ipeaks.insert(ipeaks.end(), jpeaks.begin(), jpeaks.end());

	return ipeaks;
}

std::vector<PeakInfo> findBands(
	const ArrayMap<double> & x, double peakThreshold)
{
	return findBandInRange(x, peakThreshold, 0, int(x.size()));
}

