const CODING_PROBLEMS = [
  {
    id: 1,
    title: "Two Sum",
    difficulty: "Easy",
    category: "Array",
    description: `Given an array of integers \`nums\` and an integer \`target\`, return indices of the two numbers that add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

**Example:**
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9`,
    hints: [
      "Think about using a data structure to store values you've already seen.",
      "For each number, check if (target - number) exists in your stored values.",
      "A hash map gives you O(1) lookup — store value → index."
    ],
    solution: `function twoSum(nums, target) {
  const map = new Map(); // value -> index
  
  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];
    
    if (map.has(complement)) {
      return [map.get(complement), i];
    }
    
    map.set(nums[i], i);
  }
  
  return [];
}`,
    explanation: "Use a hash map to store each number and its index. For each element, check if its complement (target - current) already exists in the map. This gives O(n) time and O(n) space."
  },
  {
    id: 2,
    title: "Valid Parentheses",
    difficulty: "Easy",
    category: "Stack",
    description: `Given a string \`s\` containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

A string is valid if:
- Open brackets must be closed by the same type of brackets.
- Open brackets must be closed in the correct order.

**Example:**
Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false`,
    hints: [
      "Think about what data structure naturally handles 'last opened, first closed'.",
      "A stack! Push opening brackets, pop when you see a closing bracket.",
      "If the top of the stack doesn't match the closing bracket — it's invalid."
    ],
    solution: `function isValid(s) {
  const stack = [];
  const map = { ')': '(', '}': '{', ']': '[' };
  
  for (const char of s) {
    if (!map[char]) {
      // Opening bracket
      stack.push(char);
    } else {
      // Closing bracket — must match top of stack
      if (stack.pop() !== map[char]) return false;
    }
  }
  
  return stack.length === 0;
}`,
    explanation: "Use a stack. Push opening brackets. For each closing bracket, pop from stack and check it matches. At end, stack must be empty. O(n) time, O(n) space."
  },
  {
    id: 3,
    title: "Reverse a Linked List",
    difficulty: "Easy",
    category: "Linked List",
    description: `Given the head of a singly linked list, reverse the list, and return the reversed list.

**Example:**
Input: 1 → 2 → 3 → 4 → 5
Output: 5 → 4 → 3 → 2 → 1`,
    hints: [
      "Think about iteratively reversing pointers one by one.",
      "You need to track: previous node, current node, and next node.",
      "For each node: save next, point current.next to prev, move prev and current forward."
    ],
    solution: `function reverseList(head) {
  let prev = null;
  let curr = head;
  
  while (curr !== null) {
    const next = curr.next; // save next
    curr.next = prev;       // reverse pointer
    prev = curr;            // move prev forward
    curr = next;            // move curr forward
  }
  
  return prev; // prev is now the new head
}`,
    explanation: "Iterative approach: maintain prev and curr pointers. At each step, reverse the link and advance both pointers. O(n) time, O(1) space."
  },
  {
    id: 4,
    title: "Maximum Subarray (Kadane's Algorithm)",
    difficulty: "Medium",
    category: "Dynamic Programming",
    description: `Given an integer array \`nums\`, find the subarray which has the largest sum and return its sum.

**Example:**
Input: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Output: 6
Explanation: [4, -1, 2, 1] has the largest sum = 6`,
    hints: [
      "Brute force is O(n²) — we can do better.",
      "At each position, decide: extend the previous subarray or start fresh?",
      "Track the max sum seen so far as you iterate."
    ],
    solution: `function maxSubArray(nums) {
  let maxSum = nums[0];
  let currentSum = nums[0];
  
  for (let i = 1; i < nums.length; i++) {
    // Either extend previous subarray or start new
    currentSum = Math.max(nums[i], currentSum + nums[i]);
    maxSum = Math.max(maxSum, currentSum);
  }
  
  return maxSum;
}`,
    explanation: "Kadane's Algorithm: at each index, the max subarray ending here is either just the current element (start fresh) or current element + previous max subarray. O(n) time, O(1) space."
  },
  {
    id: 5,
    title: "Binary Search",
    difficulty: "Easy",
    category: "Binary Search",
    description: `Given an array of integers \`nums\` sorted in ascending order, and an integer \`target\`, write a function to search target in nums. Return index if found, else return -1.

**Example:**
Input: nums = [-1, 0, 3, 5, 9, 12], target = 9
Output: 4`,
    hints: [
      "The array is sorted — don't use linear search!",
      "Compare the middle element with target to eliminate half the array each time.",
      "Keep track of left and right bounds. Update mid = Math.floor((left+right)/2)."
    ],
    solution: `function search(nums, target) {
  let left = 0;
  let right = nums.length - 1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    if (nums[mid] === target) return mid;
    else if (nums[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  
  return -1;
}`,
    explanation: "Classic binary search. Each iteration eliminates half the search space. O(log n) time, O(1) space. The key insight: if mid < target, answer is in right half; if mid > target, answer is in left half."
  },
  {
    id: 6,
    title: "Climbing Stairs",
    difficulty: "Easy",
    category: "Dynamic Programming",
    description: `You are climbing a staircase. It takes \`n\` steps to reach the top. Each time you can climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Example:**
Input: n = 3
Output: 3
Explanation: 1+1+1, 1+2, 2+1`,
    hints: [
      "Notice that to reach step n, you can come from step n-1 (1 step) or step n-2 (2 steps).",
      "ways(n) = ways(n-1) + ways(n-2). Sound familiar?",
      "It's Fibonacci! No need to store all values — just track last two."
    ],
    solution: `function climbStairs(n) {
  if (n <= 2) return n;
  
  let prev2 = 1; // ways to reach step 1
  let prev1 = 2; // ways to reach step 2
  
  for (let i = 3; i <= n; i++) {
    const curr = prev1 + prev2;
    prev2 = prev1;
    prev1 = curr;
  }
  
  return prev1;
}`,
    explanation: "This is essentially Fibonacci. ways(n) = ways(n-1) + ways(n-2). Space-optimized DP: only store last two values. O(n) time, O(1) space."
  },
  {
    id: 7,
    title: "Number of Islands",
    difficulty: "Medium",
    category: "Graph / BFS/DFS",
    description: `Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and formed by connecting adjacent lands horizontally or vertically.

**Example:**
Input:
11110
11010
11000
00000
Output: 1`,
    hints: [
      "Iterate through the grid. When you find land ('1'), that's a new island.",
      "Use DFS/BFS to mark all connected land cells as visited (turn them to '0').",
      "Count how many times you start a DFS/BFS traversal."
    ],
    solution: `function numIslands(grid) {
  let count = 0;
  
  function dfs(r, c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] === '0') return;
    grid[r][c] = '0'; // mark visited
    dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1);
  }
  
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (grid[r][c] === '1') {
        count++;
        dfs(r, c);
      }
    }
  }
  
  return count;
}`,
    explanation: "For each unvisited land cell, increment count and use DFS to mark the entire connected island as visited. O(m×n) time and space where m,n are grid dimensions."
  },
  {
    id: 8,
    title: "Merge Two Sorted Lists",
    difficulty: "Easy",
    category: "Linked List",
    description: `You are given the heads of two sorted linked lists \`list1\` and \`list2\`. Merge the two lists in a sorted order. Return the head of the merged linked list.

**Example:**
Input: list1 = 1→2→4, list2 = 1→3→4
Output: 1→1→2→3→4→4`,
    hints: [
      "Use a dummy node as the starting point to simplify the logic.",
      "Compare the current nodes of both lists. Append the smaller one.",
      "After one list is exhausted, append the remaining nodes of the other."
    ],
    solution: `function mergeTwoLists(list1, list2) {
  const dummy = { next: null };
  let curr = dummy;
  
  while (list1 && list2) {
    if (list1.val <= list2.val) {
      curr.next = list1;
      list1 = list1.next;
    } else {
      curr.next = list2;
      list2 = list2.next;
    }
    curr = curr.next;
  }
  
  curr.next = list1 || list2; // attach remaining
  return dummy.next;
}`,
    explanation: "Dummy node trick avoids edge cases. Use a pointer to build result: at each step, take the smaller node from either list. Append remaining once one list ends. O(n+m) time, O(1) space."
  }
];
