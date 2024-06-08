module ABMSimulation
    ( 
        runABM
    ) where
import AgentBased
import Control.Monad (replicateM)

-- Function to run the ABM model for a given number of runs and time steps
runABM :: Integer -> Double -> Double -> Int -> Int -> Int -> Int -> IO ([[Double]], [[Int]])
runABM n p omega b k t runs = do
    results <- replicateM runs $ do
        let model = initializeModel n p omega b k  -- Initialize the model
        finalModel <- runModel t model               -- Run the model for t steps
        let returns = dailyReturns finalModel
        let volumes = dailyTradingVolumes finalModel
        return (returns, volumes)
    let (returns, volumes) = unzip results
    return (returns, volumes)

probabilityOfTrading :: Double -> Double -> Double
probabilityOfTrading vf v = vc / (250 * 2)
  where
    vc = (v - 0.83 * vf) / (1 - 0.83)
