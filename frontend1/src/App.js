import { useState } from 'react';
import { Upload, LogOut, FileText, Download, Share2 } from 'lucide-react';

export default function App() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex justify-between items-center">
          <h1 className="text-xl font-semibold text-gray-800">SentimentAI Dashboard</h1>
          <div className="flex gap-3">
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition">
              <Upload size={16} />
              Upload Reviews
            </button>
            <button className="text-gray-600 hover:text-gray-800 px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition">
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        {/* Filters Section */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Filters & Options</h2>
          <div className="flex gap-3 items-center">
            <select className="px-4 py-2 border border-gray-300 rounded-md text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option>Filter by Date</option>
              <option>Last 7 Days</option>
              <option>Last 30 Days</option>
              <option>Last 90 Days</option>
            </select>
            <select className="px-4 py-2 border border-gray-300 rounded-md text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option>All Products</option>
              <option>Product A</option>
              <option>Product B</option>
            </select>
            <select className="px-4 py-2 border border-gray-300 rounded-md text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option>All Categories</option>
              <option>Electronics</option>
              <option>Clothing</option>
            </select>
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md text-sm font-medium transition">
              Apply Filters
            </button>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Upload Reviews Data</h2>
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition ${
              dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="mx-auto mb-4 text-gray-400" size={48} />
            <p className="text-gray-700 font-medium mb-2">Drop files here or click to upload</p>
            <p className="text-gray-500 text-sm mb-4">Support CSV, JSON, or plain text files</p>
            <div className="flex gap-3 justify-center">
              <label className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md text-sm font-medium cursor-pointer transition">
                Choose File
                <input
                  type="file"
                  className="hidden"
                  accept=".csv,.json,.txt"
                  onChange={handleFileChange}
                />
              </label>
              <button className="bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 px-6 py-2 rounded-md text-sm font-medium transition">
                Paste Text
              </button>
            </div>
            {file && (
              <p className="mt-4 text-sm text-gray-600">Selected: {file.name}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Sentiment Analysis Results */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-6">Sentiment Analysis Results</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Positive</p>
                    <p className="text-sm text-gray-500">1,247 reviews</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-green-600">68.4%</p>
                  <p className="text-xs text-gray-500">Confidence: 94%</p>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-yellow-100 rounded-full flex items-center justify-center">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Neutral</p>
                    <p className="text-sm text-gray-500">423 reviews</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-yellow-600">23.2%</p>
                  <p className="text-xs text-gray-500">Confidence: 87%</p>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">Negative</p>
                    <p className="text-sm text-gray-500">153 reviews</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-red-600">8.4%</p>
                  <p className="text-xs text-gray-500">Confidence: 91%</p>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <p className="text-sm text-gray-600">Total Reviews Analyzed</p>
                <p className="text-xl font-bold text-gray-800">1,823</p>
              </div>
            </div>
          </div>

          {/* Sentiment Distribution */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold text-gray-800">Sentiment Distribution</h2>
            </div>
            
            <div className="flex items-center justify-center mb-6">
              <svg viewBox="0 0 200 200" className="w-64 h-64">
                <circle cx="100" cy="100" r="80" fill="none" stroke="#10b981" strokeWidth="40" strokeDasharray="344 344" strokeDashoffset="0" transform="rotate(-90 100 100)" />
                <circle cx="100" cy="100" r="80" fill="none" stroke="#f59e0b" strokeWidth="40" strokeDasharray="126 344" strokeDashoffset="-344" transform="rotate(-90 100 100)" />
                <circle cx="100" cy="100" r="80" fill="none" stroke="#ef4444" strokeWidth="40" strokeDasharray="46 344" strokeDashoffset="-470" transform="rotate(-90 100 100)" />
                <circle cx="100" cy="100" r="50" fill="white" />
              </svg>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-700">Positive</span>
                </div>
                <span className="text-sm font-semibold text-gray-800">68.4%</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span className="text-sm text-gray-700">Neutral</span>
                </div>
                <span className="text-sm font-semibold text-gray-800">23.2%</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span className="text-sm text-gray-700">Negative</span>
                </div>
                <span className="text-sm font-semibold text-gray-800">8.4%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Trend Over Time */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-semibold text-gray-800">Trend Over Time</h2>
          </div>
          
          <div className="h-64 flex items-end justify-between gap-2 px-4">
            {[
              { day: 'Mon', value: 280 },
              { day: 'Tue', value: 320 },
              { day: 'Wed', value: 290 },
              { day: 'Thu', value: 340 },
              { day: 'Fri', value: 310 },
              { day: 'Sat', value: 270 },
              { day: 'Sun', value: 300 }
            ].map((item, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-2">
                <div
                  className="w-full bg-blue-500 rounded-t-lg transition-all hover:bg-blue-600"
                  style={{ height: `${(item.value / 340) * 200}px` }}
                ></div>
                <span className="text-xs text-gray-600 font-medium">{item.day}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Export Reports */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Export Reports</h2>
          <div className="flex gap-3 mb-4">
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition">
              <Download size={16} />
              Download CSV Report
            </button>
            <button className="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition">
              <FileText size={16} />
              Download PDF Report
            </button>
            <button className="bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 px-6 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition">
              <Share2 size={16} />
              Share Report
            </button>
          </div>
          <p className="text-sm text-gray-500">Reports include detailed sentiment analysis, confidence scores, and trending data.</p>
        </div>
      </div>
    </div>
  );
}