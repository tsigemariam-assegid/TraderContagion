module StochasticSimulation 
(

) where
import Stochastic
import Control.Monad (replicateM, forM_)

-- Function to run the stochastic model for a given number of runs and time steps
runStochasticModel :: Integer -> Double -> Double -> Bool -> Double -> Int -> Int -> Int -> IO [[Double]]
runStochasticModel n p init timeHorizon d m t runs = do
    results <- replicateM runs $ do
        let model = initializeStochasticModel n p init timeHorizon d m
        finalModel <- runModel t model
        return (returns finalModel)
    return results


-- main :: IO ()
-- main = do
--     let n = -- your value for n
--     let p = -- your value for p
--     let init = 10
--     let timeHorizon = -- True or False
--     let m = -- your value for M
--     let d = -- your value for d
--     let t = -- your value for t
--     let runs = -- your number of runs

--     simResults <- runStochasticModel n p init timeHorizon m d t runs

--     -- Process or print the results
--     forM_ simResults $ \runResult -> do
--         print runResult
