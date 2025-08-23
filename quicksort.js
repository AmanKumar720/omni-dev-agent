/**
 * Quicksort Algorithm Implementation in JavaScript
 * 
 * Quicksort is a highly efficient, comparison-based sorting algorithm.
 * It uses a divide-and-conquer strategy to sort elements.
 * 
 * Time Complexity:
 * - Average: O(n log n)
 * - Worst: O(n²) - when array is already sorted or reverse sorted
 * - Best: O(n log n)
 * 
 * Space Complexity: O(log n) - due to recursive call stack
 */

/**
 * Swaps two elements in an array
 * @param {Array} arr - The array to swap elements in
 * @param {number} i - First index
 * @param {number} j - Second index
 */
function swap(arr, i, j) {
    const temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

/**
 * Chooses a pivot and partitions the array around it
 * @param {Array} arr - The array to partition
 * @param {number} low - Starting index
 * @param {number} high - Ending index
 * @returns {number} - The pivot index
 */
function partition(arr, low, high) {
    // Choose the rightmost element as pivot
    const pivot = arr[high];
    
    // Index of smaller element
    let i = low - 1;
    
    // Compare each element with pivot
    for (let j = low; j < high; j++) {
        // If current element is smaller than or equal to pivot
        if (arr[j] <= pivot) {
            i++; // Increment index of smaller element
            swap(arr, i, j);
        }
    }
    
    // Place pivot in its correct position
    swap(arr, i + 1, high);
    
    return i + 1;
}

/**
 * Main quicksort function
 * @param {Array} arr - The array to sort
 * @param {number} low - Starting index (default: 0)
 * @param {number} high - Ending index (default: arr.length - 1)
 * @returns {Array} - The sorted array (mutates original array)
 */
function quicksort(arr, low = 0, high = arr.length - 1) {
    // Base case: if low is less than high
    if (low < high) {
        // Find pivot element such that
        // elements smaller than pivot are on the left
        // elements greater than pivot are on the right
        const pivotIndex = partition(arr, low, high);
        
        // Recursively sort elements before and after partition
        quicksort(arr, low, pivotIndex - 1);
        quicksort(arr, pivotIndex + 1, high);
    }
    
    return arr;
}

/**
 * Alternative quicksort implementation using functional programming approach
 * This version doesn't mutate the original array
 * @param {Array} arr - The array to sort
 * @returns {Array} - A new sorted array
 */
function quicksortFunctional(arr) {
    // Base case: arrays with 0 or 1 element are already sorted
    if (arr.length <= 1) {
        return arr;
    }
    
    // Choose pivot (middle element for better performance)
    const pivot = arr[Math.floor(arr.length / 2)];
    
    // Partition array into elements less than, equal to, and greater than pivot
    const less = arr.filter(x => x < pivot);
    const equal = arr.filter(x => x === pivot);
    const greater = arr.filter(x => x > pivot);
    
    // Recursively sort and combine
    return [...quicksortFunctional(less), ...equal, ...quicksortFunctional(greater)];
}

// Example usage and testing
function runExamples() {
    console.log("=== Quicksort Algorithm Examples ===\n");
    
    // Test 1: Basic array
    const arr1 = [64, 34, 25, 12, 22, 11, 90];
    console.log("Original array:", arr1);
    const sortedArr1 = [...arr1];
    quicksort(sortedArr1);
    console.log("Sorted array:", sortedArr1);
    console.log();
    
    // Test 2: Already sorted array
    const arr2 = [1, 2, 3, 4, 5];
    console.log("Original array:", arr2);
    const sortedArr2 = [...arr2];
    quicksort(sortedArr2);
    console.log("Sorted array:", sortedArr2);
    console.log();
    
    // Test 3: Reverse sorted array
    const arr3 = [5, 4, 3, 2, 1];
    console.log("Original array:", arr3);
    const sortedArr3 = [...arr3];
    quicksort(sortedArr3);
    console.log("Sorted array:", sortedArr3);
    console.log();
    
    // Test 4: Array with duplicates
    const arr4 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    console.log("Original array:", arr4);
    const sortedArr4 = [...arr4];
    quicksort(sortedArr4);
    console.log("Sorted array:", sortedArr4);
    console.log();
    
    // Test 5: Empty array
    const arr5 = [];
    console.log("Original array:", arr5);
    const sortedArr5 = [...arr5];
    quicksort(sortedArr5);
    console.log("Sorted array:", sortedArr5);
    console.log();
    
    // Test 6: Single element array
    const arr6 = [42];
    console.log("Original array:", arr6);
    const sortedArr6 = [...arr6];
    quicksort(sortedArr6);
    console.log("Sorted array:", sortedArr6);
    console.log();
    
    // Test 7: Functional quicksort (doesn't mutate original)
    const arr7 = [64, 34, 25, 12, 22, 11, 90];
    console.log("Original array:", arr7);
    const sortedArr7 = quicksortFunctional(arr7);
    console.log("Sorted array (functional):", sortedArr7);
    console.log("Original array unchanged:", arr7);
    console.log();
    
    // Performance test
    const largeArray = Array.from({length: 1000}, () => Math.floor(Math.random() * 1000));
    console.log("Performance test with 1000 random numbers:");
    console.log("First 10 elements:", largeArray.slice(0, 10));
    
    const startTime = performance.now();
    const sortedLargeArray = [...largeArray];
    quicksort(sortedLargeArray);
    const endTime = performance.now();
    
    console.log("Sorted in", (endTime - startTime).toFixed(2), "milliseconds");
    console.log("First 10 sorted elements:", sortedLargeArray.slice(0, 10));
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        quicksort,
        quicksortFunctional,
        partition,
        swap
    };
}

// Run examples if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    runExamples();
}