import React from "react";

function TechnologyList() {
  return (
    <div className="flex flex-col items-center justify-center">
      <h3 className="mb-5 text-lg font-medium text-gray-900 dark:text-white">
        Choose technology:
      </h3>
      <ul className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
        <li>
          <input
            type="checkbox"
            id="react-option"
            value=""
            className="hidden peer"
            required
          />
          <label
            htmlFor="react-option"
            className="inline-flex flex-col items-center justify-center w-full p-5 text-gray-500 bg-white border border-gray-200 rounded-lg cursor-pointer dark:hover:text-gray-300 dark:border-gray-700 peer-checked:border-blue-600 hover:text-gray-600 dark:peer-checked:text-gray-300 peer-checked:text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:bg-gray-800 dark:hover:bg-gray-700"
          >
            <div className="text-lg font-semibold">React Js</div>
            <div className="text-sm text-center">
              A JavaScript library for building user interfaces.
            </div>
          </label>
        </li>
        <li>
          <input
            type="checkbox"
            id="flowbite-option"
            value=""
            className="hidden peer"
          />
          <label
            htmlFor="flowbite-option"
            className="inline-flex flex-col items-center justify-center w-full p-5 text-gray-500 bg-white border border-gray-200 rounded-lg cursor-pointer dark:hover:text-gray-300 dark:border-gray-700 peer-checked:border-blue-600 hover:text-gray-600 dark:peer-checked:text-gray-300 peer-checked:text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:bg-gray-800 dark:hover:bg-gray-700"
          >
            <div className="text-lg font-semibold">Vue Js</div>
            <div className="text-sm text-center">
              Vue.js is a model–view front end JavaScript framework.
            </div>
          </label>
        </li>
        <li>
          <input
            type="checkbox"
            id="angular-option"
            value=""
            className="hidden peer"
          />
          <label
            htmlFor="angular-option"
            className="inline-flex flex-col items-center justify-center w-full p-5 text-gray-500 bg-white border border-gray-200 rounded-lg cursor-pointer dark:hover:text-gray-300 dark:border-gray-700 peer-checked:border-blue-600 hover:text-gray-600 dark:peer-checked:text-gray-300 peer-checked:text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:bg-gray-800 dark:hover:bg-gray-700"
          >
            <div className="text-lg font-semibold">Angular</div>
            <div className="text-sm text-center">
              A TypeScript-based web application framework.
            </div>
          </label>
        </li>
      </ul>
    </div>
  );
}

export default TechnologyList;
