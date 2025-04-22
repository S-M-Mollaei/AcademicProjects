This code is related to online first-person game in which the winner is the one who kills all the opponents.

Inputs: The size of the battle area, speed of each player and number of players are as inputs.

Output: game time (time to win), the number of killed opponents for the winner and the average number of killed opponents for all the players are as output in function of the inputs.

One class is created for each player in which the position, time to live and number of killed-players exist. 

Three functions are defined as follows:
1.	New position:	 every new position is assigned in way that they are in the battle area.
			 We assume that new positions are created when all the players reach their destination.

2.	Move: 	this function simulate the movement of fighters. Each player goes towards new position just 
		by selecting three adjacent points (in case speed is one cell per one movement) 
		in order to avoid players to go around the path and guarantee the convergence. 
		In case the speed (step) is greater than the distance of the initial 
		and final position (either along x axis or y axis), speed is ignored and the distance is selected as new speed along the certain axis.

3.	Fight: 	this function is defined to decide which player can live in case of encountering in specific position during movements.
	 	The winner is chosen based on random binary selection. If more than two players reach one position, just one of them can live.

Implementation: 
We define three lists of specific numbers to each input and the run the simulator until just one player lives and wins the game.
Simulating the game is performed through a loop with the condition implying that just one player can live and the game time metric is based on move per cell. 
There are nine plots of outputs based on the inputs. It can be seen that simulation is totally randomized in terms of movement and getting killed.
