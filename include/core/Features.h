#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef INCLUDE_FEATURES_H_
#define INCLUDE_FEATURES_H_

#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <unordered_map>

using namespace cv::gpu;
template<class Type>
class Features
{
public:
	Features()
	{

	}

	Features(Features<Type>& oldFeatures, std::vector<Type> newPoints,
				std::vector<unsigned char> status)
	{
		unsigned int j = 0;
		for (unsigned int i = 0; i < oldFeatures.size(); i++)
		{
			if (status[i])
			{
				unsigned int id_j = oldFeatures.getId(i);
				ids.push_back(id_j);
				indexes[id_j] = j;
				points.push_back(newPoints[i]);
				j++;
			}
		}
	}

	Features(Features<Type>& oldFeatures, std::vector<Type> newPoints,
				std::vector<int> indices)
	{
		unsigned int j = 0;
		for (auto index : indices)
		{
			unsigned int id_j = oldFeatures.getId(index);
			ids.push_back(id_j);
			indexes[id_j] = j;
			points.push_back(newPoints[index]);
			j++;
		}
	}

	Features(Features<Type>& oldFeatures, std::vector<Type> newPoints)
	{
		unsigned int j = 0;
		for (unsigned int i = 0; i < oldFeatures.size(); i++)
		{
			unsigned int id_j = oldFeatures.getId(i);
			ids.push_back(id_j);
			indexes[id_j] = j;
			points.push_back(newPoints[i]);
			j++;
		}
	}

	void addPoint(const Type& point, unsigned int id)
	{
		unsigned int index = points.size();
		points.push_back(point);
		ids.push_back(id);
		indexes[id] = index;
	}

	void addPoints(Features<Type>& features)
	{
		for (unsigned int i = 0; i < features.size(); i++)
		{
			addPoint(features[i], features.getId(i));
		}
	}

	void addPoints(Features<Type>& features, std::vector<unsigned char>& mask)
	{
		for (unsigned int i = 0; i < features.size(); i++)
		{
			if (mask[i])
				addPoint(features[i], features.getId(i));
		}
	}

	Type& operator[](unsigned int i)
	{
		return points[i];
	}

	const Type& operator[](unsigned int i) const
	{
		return points[i];
	}

	typename std::vector<Type>::iterator begin()
	{
		return points.begin();
	}

	typename std::vector<Type>::iterator end()
	{
		return points.end();
	}

	inline std::vector<Type>& getPoints()
	{
		return points;
	}

	inline unsigned int getId(unsigned int index) const
	{
		return ids[index];
	}

	inline unsigned int getIndex(unsigned int id)
	{
		if (indexes.count(id) == 0)
			throw std::runtime_error("no index");
		return indexes[id];
	}

	inline size_t size() const
	{
		return points.size();
	}

	inline bool contains(unsigned int id) const
	{
		assert(ids.size() == points.size() && points.size() == indexes.size());
		return indexes.count(id) != 0;
	}

	void scalePoints(double scale)
	{
		for (auto& p : points)
			p *= scale;
	}

private:
	std::vector<Type> points;
	std::vector<unsigned int> ids;
	std::unordered_map<unsigned int, unsigned int> indexes;

};

typedef Features<cv::Point2f> Features2D;
typedef Features<cv::Point3f> Features3D;

#endif /* INCLUDE_FEATURES_H_ */
